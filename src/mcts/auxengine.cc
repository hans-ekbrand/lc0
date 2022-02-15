/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2022 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "mcts/search.h"

#include <algorithm>
#include <boost/process.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "utils/fastmath.h"
#include "utils/random.h"

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(1,0);

namespace lczero {

void Search::OpenAuxEngine() REQUIRES(threads_mutex_) {
  if (params_.GetAuxEngineFile() == "") return;
  for(int i = 0; i < params_.GetAuxEngineInstances(); i++){
    auxengine_threads_.emplace_back([this]() { AuxEngineWorker(); });
  }
}

// void SearchWorker::AuxMaybeEnqueueNode(Node* n, int source) {
void SearchWorker::AuxMaybeEnqueueNode(Node* n) {
  // the caller (DoBackupUpdate()->DoBackupUpdateSingleNode()) has a lock on search_->nodes_mutex_, so no other thread will change n right now.
  // There are two callers, also PreExtend() which also has that lock

  // Since we take a lock below, have to check if search is stopped.
  // No, the real reason is that we must not add nodes after purging has started.
  if (search_->stop_.load(std::memory_order_acquire)){
    return;
  }

  search_->search_stats_->auxengine_mutex_.lock();

  search_->number_of_times_called_AuxMaybeEnqueueNode_ += 1; // only for stats, not functionally necessary.
  
  // if purging has already happened, then do nothing
  if(! search_->search_stats_->final_purge_run) {
    // if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE
    //   << "AuxMaybeEnqueueNode() picked node: " << n->DebugString() 
    //   << " for the persistent_queue_of_nodes which has size: "
    //   << search_->search_stats_->persistent_queue_of_nodes.size()
    //   << " The source was " << source;
    n->SetAuxEngineMove(0xfffe); // magic for pending
    if(search_->search_stats_->persistent_queue_of_nodes.size() < 15000) { // safety net for too low values of AuxEngineThreshold, which would cause this queue to overflow somehow, or just take too much time to check between moves.
      search_->search_stats_->persistent_queue_of_nodes.push(n);
      // search_->search_stats_->source_of_queued_nodes.push(source);
      search_->auxengine_cv_.notify_one();
    }
  }
  search_->search_stats_->auxengine_mutex_.unlock();
}

void Search::AuxEngineWorker() {

  // aquire a lock on pure_stats_mutex_ to ensure no other thread is
  // modifying search_stats_->thread_counter or the vector_of_*
  // vectors

  search_stats_->pure_stats_mutex_.lock();

  // Find out which thread we are by reading the thread_counter.

  // Don't increment the thread_counter before all global vectors are
  // initiated, or MaybeTriggerStop() in search.cc will try to write
  // to uninitiated adresses.

  long unsigned int our_index = search_stats_->thread_counter;

  // If we are the first thread, and the final purge already has taken place, then return immediately.
  // search_stats_->final_purge_run is protected by search_stats_->auxengine_mutex_.

  search_stats_->auxengine_mutex_.lock();    
  if(our_index == 0 && search_stats_->final_purge_run){
    LOGFILE << "AuxEngineWorker() Thread 0 returning early because purge as already taken place";
    search_stats_->auxengine_mutex_.unlock();    
    search_stats_->pure_stats_mutex_.unlock();    
    return;
  }
  search_stats_->auxengine_mutex_.unlock();
  
  // Also, if search has stopped, do not spawn a another helper instance until the next move.
  if(stop_.load(std::memory_order_acquire)){
    LOGFILE << "AuxEngineWorker() Thread 0 returning early because search has already stopped.";
    search_stats_->pure_stats_mutex_.unlock();
    return;
  }
  
  // if our_index is greater than the size of the vectors then we know for sure we must start/initiate everything.
  // if our_index + 1 is equal to, or smaller than the size of the vectors then we can safely check search_stats_->vector_of_auxengine_ready_[our_index] and act if it is false

  if(our_index + 1 > search_stats_->vector_of_auxengine_ready_.size() ||
     (
      our_index + 1 <= search_stats_->vector_of_auxengine_ready_.size() &&
      ! search_stats_->vector_of_auxengine_ready_[our_index]
     )
   ) {

    // increase the thread_counter.
    search_stats_->thread_counter++;
 
    // populate the global vectors. 
    search_stats_->vector_of_ipstreams.emplace_back(new boost::process::ipstream);
    search_stats_->auxengine_stopped_mutex_.lock();
    search_stats_->vector_of_opstreams.emplace_back(new boost::process::opstream);
    search_stats_->auxengine_stopped_mutex_.unlock();
    search_stats_->vector_of_children.emplace_back(new boost::process::child);

    // Start the helper
    *search_stats_->vector_of_children[our_index] = boost::process::child(params_.GetAuxEngineFile(), boost::process::std_in < *search_stats_->vector_of_opstreams[our_index], boost::process::std_out > *search_stats_->vector_of_ipstreams[our_index]);

    // Record that we have started, so that we can skip this on the next invocation.
    search_stats_->vector_of_auxengine_ready_.push_back(true);

    // unlock while we wait for the engine to be finished?
    search_stats_->pure_stats_mutex_.unlock();

    search_stats_->auxengine_stopped_mutex_.lock();
    search_stats_->auxengine_stopped_.push_back(true);
    search_stats_->auxengine_stopped_mutex_.unlock();

    {
      std::string bar;
      // If AuxEngineOptionsOnRoot is set, Thread zero uses a different parameter and it continuosly explores root node only.
      // If not set, thread zero becomes just another in-tree helper instance.
      if(our_index == 0 &&
	 !params_.GetAuxEngineOptionsOnRoot().empty()
	 ){
	bar = params_.GetAuxEngineOptionsOnRoot();
      } else {
	// in-tree time based evaluations
	bar = params_.GetAuxEngineOptions();
      }
      std::istringstream iss(bar);
      std::string token;
      while(std::getline(iss, token, '=')) {
        std::ostringstream oss;
        oss << "setoption name " << token;
        std::getline(iss, token, ';');
        oss << " value " << token;
	search_stats_->auxengine_stopped_mutex_.lock();
	*search_stats_->vector_of_opstreams[our_index] << oss.str() << std::endl;
	search_stats_->auxengine_stopped_mutex_.unlock();	
      }
      search_stats_->auxengine_stopped_mutex_.lock();
      *search_stats_->vector_of_opstreams[our_index] << "uci" << std::endl;
      search_stats_->auxengine_stopped_mutex_.unlock();      
    }
    std::string line;
    while(std::getline(*search_stats_->vector_of_ipstreams[our_index], line)) {
      if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << line;
      std::istringstream iss(line);
      std::string token;
      iss >> token >> std::ws;
      if (token == "uciok") {
        break;
      } else if (token == "option") {
        iss >> token >> std::ws;
        if (token == "name") {
          iss >> token >> std::ws;
          if (token == "SyzygyPath" && syzygy_tb_) {
            std::ostringstream oss;
            oss << "setoption name SyzygyPath value " << syzygy_tb_->get_paths();
            if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << oss.str();
	    search_stats_->auxengine_stopped_mutex_.lock();
	    *search_stats_->vector_of_opstreams[our_index] << oss.str() << std::endl;	    
	    search_stats_->auxengine_stopped_mutex_.unlock();
          }
        }
      }
    }

    search_stats_->pure_stats_mutex_.lock();
    if(our_index == 0){
      // Initiate some stats and parameters (Threshold needs to be set
      // earlier, see search() in search.cc)
      search_stats_->AuxEngineTime = params_.GetAuxEngineTime();
      search_stats_->Number_of_nodes_added_by_AuxEngine = 0;
      search_stats_->Total_number_of_nodes = 0;
      search_stats_->initial_purge_run = true;
      search_stats_->my_pv_cache_mutex_.lock();      
      search_stats_->my_pv_cache_.clear(); // Clear the PV cache.
      search_stats_->my_pv_cache_mutex_.unlock();      
      if(search_stats_->New_Game){
	search_stats_->New_Game = false;
	// Automatically inactivate the queueing machinery if there is only one instance AND OptionsOnRoot is NON-empty. Could save some time in ultra-bullet.
	if(params_.GetAuxEngineInstances() == 1 &&
	   !params_.GetAuxEngineOptionsOnRoot().empty()
	   ){
	  search_stats_->AuxEngineThreshold = 0;
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Inactivating the queueing machinery since there is exactly one instance and OnRoot is non-empty.";
	} else  {
	  search_stats_->AuxEngineThreshold = params_.GetAuxEngineThreshold();
	}
      }
    }
  } else {

    // AuxEngine(s) were already started. If we are thread zero then (1) Purge the queue(s) and (2) kickstart root if the queue is empty.
    search_stats_->thread_counter++;

    if(our_index == 0){

      if(search_stats_->New_Game){
	search_stats_->AuxEngineTime = params_.GetAuxEngineTime();
	// Automatically inactivate the queueing machinery if there is only one instance and OptionsOnRoot is non-empty. Could save some time in ultra-bullet.
	if(params_.GetAuxEngineInstances() == 1 &&
	   !params_.GetAuxEngineOptionsOnRoot().empty()
	   ){
	  search_stats_->AuxEngineThreshold = 0;
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Inactivating the queueing machinery since there is exactly one instance and OnRoot is non-empty.";
	} else  {
	  search_stats_->AuxEngineThreshold = params_.GetAuxEngineThreshold();
	}
	search_stats_->Total_number_of_nodes = 0;
	search_stats_->Number_of_nodes_added_by_AuxEngine = 0;
	search_stats_->size_of_queue_at_start = 0;

	search_stats_->New_Game = false;

	// change lock to purge queue of PVs
	search_stats_->pure_stats_mutex_.unlock();
	search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock();
	search_stats_->fast_track_extend_and_evaluate_queue_ = {};
	search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();

	// different lock for queue of nodes
	search_stats_->auxengine_mutex_.lock();
	search_stats_->persistent_queue_of_nodes = {};

      } else {
	// aquire the right lock
	search_stats_->pure_stats_mutex_.unlock();
	search_stats_->auxengine_mutex_.lock();
      }
    
      // purge obsolete nodes in the queue, if any. The even elements are the actual nodes, the odd elements is root if the preceding even element is still a relevant node.
      if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "search_stats_->size_of_queue_at_start:" << search_stats_->size_of_queue_at_start;
      if(search_stats_->final_purge_run){
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Either we are not the first thread, or there is an unexpected order of execution, and final purging has already taken place. In either case not purging now.";
      } else {
	if(search_stats_->size_of_queue_at_start > 0){
	  int number_of_nodes_before_purging = int(search_stats_->size_of_queue_at_start / 2);
	  std::queue<Node*> persistent_queue_of_nodes_temp_;
	  for(int i=0; i < search_stats_->size_of_queue_at_start; i = i + 2){
	    Node * n = search_stats_->persistent_queue_of_nodes.front();
	    search_stats_->persistent_queue_of_nodes.pop();
	    Node * n_parent = search_stats_->persistent_queue_of_nodes.front();
	    search_stats_->persistent_queue_of_nodes.pop();
	    if(n_parent == root_node_){
	      // node is still relevant
	      persistent_queue_of_nodes_temp_.push(n);
	    }
	  }
	  // update search_stats_->persistent_queue_of_nodes
	  int my_size = persistent_queue_of_nodes_temp_.size();
	  for(int i=0; i < my_size; i++){      
	    search_stats_->persistent_queue_of_nodes.push(persistent_queue_of_nodes_temp_.front());
	    persistent_queue_of_nodes_temp_.pop();
	  }
	  if (params_.GetAuxEngineVerbosity() >= 4)
	    LOGFILE << "Purged " << number_of_nodes_before_purging - search_stats_->persistent_queue_of_nodes.size()
		    << " nodes from the query queue due to the move selected by the opponent. " << search_stats_->persistent_queue_of_nodes.size()
		    << " nodes remain in the queue.";
	}

	// Also purge obsolete PV:s, but that requires a different lock
	search_stats_->auxengine_mutex_.unlock();
	search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock();

	nodes_mutex_.lock_shared();
	// LOGFILE << "AuxWorker() Thread 0 succeeded in switching locks";
	// Not sure if there is guarantee this edge is still present
	// LOGFILE << "AuxEngineWorker() testing for nullptr for root node: " << root_node_->DebugString();
	bool root_valid_move_found = false;
	Move valid_move;	
	if(root_node_->GetOwnEdge() == nullptr) {
	  nodes_mutex_.unlock_shared();	  
	  LOGFILE << "AuxEngineWorker() found nullptr at the edge to the root_node_";
	  search_stats_->fast_track_extend_and_evaluate_queue_ = {};
	} else {
	  valid_move = root_node_->GetOwnEdge()->GetMove();
	  nodes_mutex_.unlock_shared();
	  root_valid_move_found = true;
	}
	
	if(root_valid_move_found &&
	   search_stats_->fast_track_extend_and_evaluate_queue_.size() > 0){
	  std::queue<std::vector<Move>> fast_track_extend_and_evaluate_queue_temp_;
	  long unsigned int my_size = search_stats_->fast_track_extend_and_evaluate_queue_.size();
	  while(search_stats_->fast_track_extend_and_evaluate_queue_.size() > 0){
	    std::vector<Move> pv = search_stats_->fast_track_extend_and_evaluate_queue_.front();
	    search_stats_->fast_track_extend_and_evaluate_queue_.pop();
	    if(pv.size() > 1){
	      // LOGFILE << "will test if pv0 is the valid_move: ";
	      if(pv[0] == valid_move){
		// remove the first move, which is the move the opponent made that lead to the current position
		// LOGFILE << "AuxEngineWorker() trying to erase the first move, size is " << pv.size();
		pv.erase(pv.begin());
		fast_track_extend_and_evaluate_queue_temp_.push(pv);
	      } else {
		// LOGFILE << pv[0].as_string() << " is not equal to " << root_node_->GetOwnEdge()->GetMove().as_string();
	      }
	    } else {
	      LOGFILE << "AuxEngineWorker() found PV of size less than 2, discarding it." << pv.size();		  
	    }
	  }
	  // Empty the queue and copy back the relevant ones.
	  search_stats_->fast_track_extend_and_evaluate_queue_ = {};
	  long unsigned int size_kept = fast_track_extend_and_evaluate_queue_temp_.size();
	  for(long unsigned int i=0; i < size_kept; i++){
	    search_stats_->fast_track_extend_and_evaluate_queue_.push(fast_track_extend_and_evaluate_queue_temp_.front());
	    fast_track_extend_and_evaluate_queue_temp_.pop();
	  }
	  if (params_.GetAuxEngineVerbosity() >= 4)	  
	  LOGFILE << "Purged " << my_size - size_kept << " PVs due to the move selected by the opponent. " << size_kept
		  << " PVs remain in the queue.";
	}
	// switch back locks.
	search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();
	search_stats_->auxengine_mutex_.lock();
	
      } // end of if(search_stats_->final_purge_run)

      // switch back the locks
      search_stats_->auxengine_mutex_.unlock();
      search_stats_->pure_stats_mutex_.lock();

      // More stuff for thread zero only
      search_stats_->initial_purge_run = true; // Inform other threads that they should not purge.
      if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxEngineWorker() finished purging/initiating, will now check if root can be queued";
      
    } // Thread zero
  } // Not starting from scratch

  search_stats_->pure_stats_mutex_.unlock();
    
  Node* n;
  bool not_yet_notified = true;
  bool root_is_queued = false;
  while (!stop_.load(std::memory_order_acquire)) {
    // if we are thread zero, don't read from the queue, just take the root node.
    if(our_index == 0 && !params_.GetAuxEngineOptionsOnRoot().empty()){
      // kickstart with the root node, no need to wait for it to get some
      // amount of visits. Except if root is not yet expanded, or lacks
      // edges for any other reason (e.g. being terminal), in which case
      // we should wait and try again later.
      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "AuxEngineWorker() Thread 0 about to aquire a shared lock nodes_mutex_ in order to read root";
      nodes_mutex_.lock_shared(); // only neede to read GetNumEdges(), SetAuxEngineMove(0xfffe) is already protected by search_stats_->auxengine_mutex_.lock();
      if(root_node_->GetNumEdges() > 0){
	// root is extended.
	search_stats_->auxengine_mutex_.lock();
	root_node_->SetAuxEngineMove(0xfffe); // mark root as pending and queue it
	nodes_mutex_.unlock_shared(); // unlock the read-lock on noodes.
	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "AuxEngineWorker() Thread 0 released shared lock nodes_mutex_.";
    	// search_stats_->source_of_queued_nodes.push(3); // inform DoAuxEngine() -> where this node came from.
	search_stats_->auxengine_mutex_.unlock(); // We will be in DoAuxEngine() until search is stopped, so unlock first.
	DoAuxEngine(root_node_, our_index);
      } else {
	nodes_mutex_.unlock_shared(); // unlock, nothing more to do until root gets edges.
	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "AuxEngineWorker() Thread 0 released shared lock nodes_mutex_.";	
	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "AuxEngineWorker() Thread 0 found root node has no edges will sleep 100 ms";
	using namespace std::chrono_literals;
	std::this_thread::sleep_for(100ms);
      }
    } else {
      // Not thread 0, or empty OnRoot options
      if(not_yet_notified){
	// Wait for search_stats_->initial_purge_run == true before starting to work.
	search_stats_->pure_stats_mutex_.lock();	
	if(!search_stats_->initial_purge_run) {
	  search_stats_->pure_stats_mutex_.unlock();
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxEngineWorker() Thread " << our_index << " waiting for thread 0 to purge the queue, will sleep 5 ms";
	  using namespace std::chrono_literals;
	  std::this_thread::sleep_for(5ms);
	} else {
	  // purge is done, just release the lock.
	  search_stats_->pure_stats_mutex_.unlock();
	}
	// OK, we are good to go.
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxEngineWorker() Thread: " << our_index << " entered main loop.";
	not_yet_notified = false;
      }

      // // You may only listen if you have this lock: auxengine_listen_mutex_ this way we avoid spurios awakenings.
      // auxengine_listen_mutex_.lock();

      // If we are thread 0, (this implies OnRoot is empty) why not kickstart with queueing the root node for a time limited query? Probably an unsual use case, but why not?
      // Kickstart root if empty OnRoot options START
      if(our_index == 0 && !root_is_queued){
	if (params_.GetAuxEngineVerbosity() >= 3) LOGFILE << "AuxEngineWorker() Thread 0 adding root node to the queue because AuxEngineOptionsOnRoot is empty.";
	nodes_mutex_.lock_shared(); // only neede to read GetNumEdges(), SetAuxEngineMove(0xfffe) is already protected by search_stats_->auxengine_mutex_.lock();
	if(root_node_->GetNumEdges() > 0){
	  nodes_mutex_.unlock_shared(); // unlock the read-lock on noodes.	  
	  // root is extended, we can enqueue it
	  // This is exactly what SearchWorker::AuxMaybeEnqueueNode() does, but we are in class Search:: now, so that function is not available.
	  search_stats_->auxengine_mutex_.lock();
	  if(! search_stats_->final_purge_run && // Note that final purge may already have happened.
	     search_stats_->persistent_queue_of_nodes.size() < 15000) { // safety net for too low values of AuxEngineThreshold, which would cause this queue to overflow somehow, or just take too much time to check between moves.
	    search_stats_->persistent_queue_of_nodes.push(root_node_);
	    auxengine_cv_.notify_one();
	  }
	  search_stats_->auxengine_mutex_.unlock();
	  root_is_queued = true;
	} else {
	  nodes_mutex_.unlock_shared(); // unlock, nothing more to do until root gets edges.
	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "AuxEngineWorker() Thread 0 found root node has no edges will sleep 100 ms";
	  using namespace std::chrono_literals;
	  std::this_thread::sleep_for(100ms);
	}
      }
      // Kickstart root if empty OnRoot options STOP

      {
	std::unique_lock<std::mutex> lock(search_stats_->auxengine_mutex_);
	// Wait until there's some work to compute.
	auxengine_cv_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) || !search_stats_->persistent_queue_of_nodes.empty(); });
	// at this point, the lock is released and aquired again, which is why we want the outer lock, without which another thread could intercept us here.
	if (stop_.load(std::memory_order_acquire)) {
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxWorker(), Thread " << our_index << " caught a stop signal while waiting for a node to process, will exit the while loop now.";
	  // auxengine_listen_mutex_.unlock();
	  search_stats_->pure_stats_mutex_.lock();
	  search_stats_->thread_counter--;
	  // Almost always log the when the last thread exits.
	  if(search_stats_->thread_counter == 0){
	    if (params_.GetAuxEngineVerbosity() >= 1) LOGFILE << "All AuxEngineWorker threads are now idle";
	  } else {
	    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxEngineWorker Thread " << our_index << " done. The thread counter is now " << search_stats_->thread_counter;
	  }
	  search_stats_->pure_stats_mutex_.unlock();
	  return;
	}
	n = search_stats_->persistent_queue_of_nodes.front();
	search_stats_->persistent_queue_of_nodes.pop();
      } // implictly release the lock on search_stats_->auxengine_mutex_
      // auxengine_listen_mutex_.unlock();
      DoAuxEngine(n, our_index);    
    } // end of not thread zero
  } // end of while loop

  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxWorker(), Thread " << our_index << " caught a stop signal after returning from DoAuxEngine(), will exit the while loop now.";
  // Decrement the thread counter so that purge in search.cc does not start before all threads are done.
  search_stats_->pure_stats_mutex_.lock();  
  search_stats_->thread_counter--;
  // Almost always log the when the last thread exits.
  if(search_stats_->thread_counter == 0){
    if (params_.GetAuxEngineVerbosity() >= 1) LOGFILE << "All AuxEngineWorker threads are now idle";
  } else {
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxEngineWorker Thread " << our_index << " done. The thread counter is now " << search_stats_->thread_counter;
  }
  search_stats_->pure_stats_mutex_.unlock();
}

  void Search::AuxEncode_and_Enqueue(std::string pv_as_string, int depth, ChessBoard my_board, Position my_position, std::vector<lczero::Move> my_moves_from_the_white_side, int source, bool require_some_depth, int thread) {
  // Take a string recieved from a helper engine, turn it into a vector with elements of type Move and queue that vector.

  // Quit early if search has stopped
 if(stop_.load(std::memory_order_acquire)) {
   // if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Would have quit early from AuxEncode_and_Enqueue() since search has stopped, but decided to take the risk and go on.";   
   if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << thread << ": Quitting early from AuxEncode_and_Enqueue() since search has stopped.";
   return;
 }

  std::istringstream iss(pv_as_string);  
  std::string pv;
  std::vector<uint16_t> pv_moves;

  std::string s = "position fen " + GetFen(my_position); // for informational purposes only.
  std::string token;

  // When we internally use the moves to extend nodes in the search tree, always use move as seen from the white side.
  // Apply the moves in reversed order to get the proper board state from which we can then make moves in legacy format.
  // std::vector<lczero::Move> my_moves;
  // std::vector<lczero::Move> my_moves_from_the_white_side;  
    
  bool flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);

  // auto bestmove_packed_int = Move(token, !flip).as_packed_int();
  // depth is distance between root and the starting point for the
  // auxengine.
  // depth_reached records the depth the helper claim to have search.
  // The PV is capped at this length (and can be shortened again in PreExt..()

  int pv_length = 1;
  int depth_reached = 0;
  int nodes_to_support = 0;
  int max_pv_length = 99; // Dirty work around for too many levels of recursion. Probably not doing anything at all.

  while(iss >> pv >> std::ws) {
    if (pv == "info"){
      continue;
    }
    if (pv == "string"){
      // not for us.
      return;
    }
    if (pv == "depth") {
      // Figure out which depth was reached (can be zero).
      iss >> depth_reached >> std::ws;
      // // Save time by ignoring PVs with low depth.
      // if(require_some_depth && depth_reached < 15) return;
    }
    if (pv == "nodes") {
      // Figure out how many nodes this PV is based on.
      iss >> nodes_to_support >> std::ws;
      // Save time by ignoring PVs with low support.
      // if(nodes_to_support < 10000){
      // 	return;
      // }
    }

    // Either "don't require depth" or depth > 14 or at least 10000 nodes
    if (pv == "pv" && (!require_some_depth || nodes_to_support >= 1000 || depth_reached > 10)) {
      while(iss >> pv >> std::ws &&
	    pv_length < depth_reached &&
	    pv_length < max_pv_length) {
	Move m;
	if (!Move::ParseMove(&m, pv, !flip)) {		
	  if (params_.GetAuxEngineVerbosity() >= 1) LOGFILE << "Thread: " << thread << " Ignoring bad pv move: " << pv;
	  break;
	  // why not return instead of break?
	}
	
	// m is always from the white side, pv is not. No need to mirror the board then? Actually, yes.

	// convert to Modern encoding, update the board and the position

	Move m_in_modern_encoding = my_board.GetModernMove(m);
	my_moves_from_the_white_side.push_back(m_in_modern_encoding); // Add the PV to the queue 
	pv_moves.push_back(m_in_modern_encoding.as_packed_int());
	my_position = Position(my_position, m_in_modern_encoding);	
	my_board.ApplyMove(m_in_modern_encoding);
	my_board.Mirror();	

	flip = !flip;
	pv_length++;
      }
    }
  }

  // Too short PV are probably not reliable (> 4 seems to suffice), too high bar can be bad with low values of AuxEngineTime
  int min_pv_size = 4;
  if (pv_moves.size() >= min_pv_size){

    // check if the PV is new
    std::ostringstream oss;
    // Convert all but the last element to avoid a trailing "," https://stackoverflow.com/questions/8581832/converting-a-vectorint-to-string
    std::copy(pv_moves.begin(), pv_moves.end()-1, std::ostream_iterator<int>(oss, ","));
    // Now add the last element with no delimiter
    oss << pv_moves.back();
    // TODO protect the PV cache with a mutex? Stockfish does not, and worst case scenario is that the same PV is sent again, so probably not needed.
    // https://stackoverflow.com/questions/8581832/converting-a-vectorint-to-string
    search_stats_->my_pv_cache_mutex_.lock();
    if ( search_stats_->my_pv_cache_.find(oss.str()) == search_stats_->my_pv_cache_.end() ) {
      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << thread << " string not found in the cache, adding it.";
      search_stats_->my_pv_cache_[oss.str()] = true;
    } else {
      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << thread << " string found in the cache. Return early.";
      search_stats_->my_pv_cache_mutex_.unlock();
      return;
    }
    search_stats_->my_pv_cache_mutex_.unlock();
    
    if (params_.GetAuxEngineVerbosity() >= 9){
      std::string debug_string;
      // No lock required here, my_moves_from_the_white_side is only a simple queue of Moves, it has nothing to do with the searchtree.
      for(int i = 0; i < (int) my_moves_from_the_white_side.size(); i++){
	debug_string = debug_string + my_moves_from_the_white_side[i].as_string() + " ";
      }
      if(played_history_.IsBlackToMove()){
	LOGFILE << "Thread: " << thread << " debug info: length of PV given to helper engine: " << depth << " position given to helper: " << s << " black to move at root, length of my_moves_from_the_white_side " << my_moves_from_the_white_side.size() << " my_moves_from_the_white_side: " << debug_string;
      } else {
	LOGFILE << "Thread: " << thread << " debug info: length of PV given to helper engine: " << depth << " position given to helper: " << s << " white to move at root, length of my_moves_from_the_white_side " << my_moves_from_the_white_side.size() << " my_moves_from_the_white_side: " << debug_string;
      }
    }

    // Because some threads doesn't shut down fast enough, e.g. because they are loading endgame databases.
    // If final purge has already taken place, then discard this PV
    long unsigned int size;
    LOGFILE << "Thread: " << thread << " Trying to get a lock on search_stats_->fast_track_extend_and_evaluate_queue_mutex_";
    search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock(); // lock this queue before starting to modify it
    LOGFILE << "Thread: " << thread << " aquired a lock on search_stats_->fast_track_extend_and_evaluate_queue_mutex_";    
    size = search_stats_->fast_track_extend_and_evaluate_queue_.size();
    if(size < 20000){ // safety net, silently drop PV:s if we cannot extend nodes fast enough. lc0 stalls when this number is too high.
      search_stats_->fast_track_extend_and_evaluate_queue_.push(my_moves_from_the_white_side);
      search_stats_->starting_depth_of_PVs_.push(depth);
      search_stats_->amount_of_support_for_PVs_.push(nodes_to_support);
      search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();
      // search_stats_->source_of_PVs.push(source);
    } else {
      if (params_.GetAuxEngineVerbosity() >= 3) LOGFILE << "Thread: " << thread << " Silently discarded a PV starting at depth " << depth << " with " << nodes_to_support  << " nodes to support it. Queue has size: " << size;
      // just unlock
      search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();	
    }
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << thread << " Added a PV starting at depth " << depth << " with " << nodes_to_support  << " nodes to support it. Queue has size: " << size;
  } else {
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << thread << " Ignoring pv because it not longer than " << min_pv_size;
  }
}

void Search::DoAuxEngine(Node* n, int index){
  // before trying to take a lock on nodes_mutex_, always check if search has stopped, in which case we return early
  if(stop_.load(std::memory_order_acquire)) {
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine, Thread " << index << " caught a stop signal beforing doing anything.";
    return;
  }

  if (params_.GetAuxEngineVerbosity() >= 9){
    nodes_mutex_.lock_shared();    
    LOGFILE << "Thread: " << index << ". DoAuxEngine() aquired a lock on nodes_ and was called for node" << n->DebugString();
    nodes_mutex_.unlock_shared();
    LOGFILE << "Thread: " << index << ". DoAuxEngine() released a lock on nodes_";
  }

  // Calculate depth.
  int depth = 0;
  if(n != root_node_){
    if(stop_.load(std::memory_order_acquire)) {
      if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Thread: " << index << " DoAuxEngine caught a stop signal before starting to calculate depth.";
      return;
    }
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " DoAuxEngine() trying to aquire a shared lock on nodes_";    
    nodes_mutex_.lock_shared();
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " DoAuxEngine() aquired a lock on nodes_";
    for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      depth++;
    }
    nodes_mutex_.unlock_shared();
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " DoAuxEngine() released a lock on nodes_";    
  }

  search_stats_->auxengine_mutex_.lock();
  // Never add nodes to the queue after search has stopped or final purge is run
  if(stop_.load(std::memory_order_acquire) ||
     search_stats_->final_purge_run){
    // just pop the source node, and unset pending so that the node can get picked up during next search.
    // search_stats_->source_of_queued_nodes.pop();
    n->SetAuxEngineMove(0xffff);
    search_stats_->auxengine_mutex_.unlock();
    return;
  }

  if (search_stats_->persistent_queue_of_nodes.size() > 0){ // if there is no node in the queue then accept unconditionally.
    if(depth > 0 &&
       depth > params_.GetAuxEngineMaxDepth()
       ){
      // Only generate a random sample if these parameters are true, save a few random samples
      if(float(1.0f)/(depth) < distribution(generator)){
	// This is exactly what SearchWorker::AuxMaybeEnqueueNode() does, but we are in class Search:: now, so that function is not available.
	// int source = search_stats_->source_of_queued_nodes.front();
	// search_stats_->source_of_queued_nodes.pop();
	search_stats_->persistent_queue_of_nodes.push(n);
	// search_stats_->source_of_queued_nodes.push(source);
	auxengine_cv_.notify_one(); // unnecessary?
	search_stats_->auxengine_mutex_.unlock();
	return;
      }
    }
  }

  // while we have this lock, also read the current value of search_stats_->AuxEngineTime, which is needed later
  int AuxEngineTime = search_stats_->AuxEngineTime;
  
  search_stats_->auxengine_mutex_.unlock();  
  
  if(depth > 0 &&
     depth > params_.GetAuxEngineMaxDepth()){
    // if (params_.GetAuxEngineVerbosity() >= 6) LOGFILE << "DoAuxEngine processing a node with high depth: " << " since sample " << sample << " is less than " << float(1.0f)/(depth);
  }
    
  // if (params_.GetAuxEngineVerbosity() >= 6) LOGFILE << "DoAuxEngine processing a node with depth: " << depth;

  std::string s = "";
  bool flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);

  // To get the moves in UCI format, we have to construct a board, starting from root and then apply the moves.
  // Traverse up to root, and store the moves in a vector.
  // When we internally use the moves to extend nodes in the search tree, always use move as seen from the white side.
  // Apply the moves in reversed order to get the proper board state from which we can then make moves in legacy format.
  std::vector<lczero::Move> my_moves;
  std::vector<lczero::Move> my_moves_from_the_white_side;  

  if(n != root_node_){
    if(stop_.load(std::memory_order_acquire)) {
      if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine caught a stop signal while populating my_moves.";
      return;
    }
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " DoAuxEngine() trying to aquire a lock on nodes_ in order to create the position for the helper.";    
    nodes_mutex_.lock_shared();  
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " DoAuxEngine() aquired a lock on nodes_ in order to create the position for the helper.";
    for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      my_moves.push_back(n2->GetOwnEdge()->GetMove(flip));
      my_moves_from_the_white_side.push_back(n2->GetOwnEdge()->GetMove());
      flip = !flip;
    }
    nodes_mutex_.unlock_shared();
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " DoAuxEngine() released a lock on nodes_.";
  }

  // Reverse the order
  std::reverse(my_moves.begin(), my_moves.end());
  std::reverse(my_moves_from_the_white_side.begin(), my_moves_from_the_white_side.end());
    
  ChessBoard my_board = played_history_.Last().GetBoard();
  Position my_position = played_history_.Last();

  // modern encoding
  for(auto& move: my_moves) {
    if (my_board.flipped()) move.Mirror();
    my_board.ApplyMove(move);
    my_position = Position(my_position, move);
    if (my_board.flipped()) move.Mirror();
    s = s + move.as_string() + " ";  // only for debugging
    my_board.Mirror();
  }

  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " add pv=" << s << " from root position: " << GetFen(played_history_.Last()); // not given to the helper engine, so modern encoding is fine here.
  s = "position fen " + GetFen(my_position);
  
  // 1. Only start the engines if we can aquire the auxengine_stopped_mutex
  // 2. Only send anything to the engines if we have aquired that mutex

  search_stats_->auxengine_stopped_mutex_.lock();  
  // Before starting, test if stop_ is set
  if (stop_.load(std::memory_order_acquire)) {
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Thread: " << index << " DoAuxEngine caught a stop signal before querying the helper.";
    search_stats_->auxengine_stopped_mutex_.unlock();
    return;
  }
  *search_stats_->vector_of_opstreams[index] << s << std::endl;
  auto auxengine_start_time = std::chrono::steady_clock::now();
  bool infinite_exploration = false;
  if(index == 0 &&
     !params_.GetAuxEngineOptionsOnRoot().empty()){
    infinite_exploration = true;
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " Starting infinite query from root node for thread 0 using the opstream at: " << &search_stats_->vector_of_opstreams[index];
    *search_stats_->vector_of_opstreams[index] << "go infinite " << std::endl;
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " Started infinite query from root node for thread 0 using the opstream at: " << &search_stats_->vector_of_opstreams[index];    
  } else {
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Thread: " << index << " Starting time limited query for thread " << index << " using the opstream at: " << &search_stats_->vector_of_opstreams[index];    
    *search_stats_->vector_of_opstreams[index] << "go movetime " << AuxEngineTime << std::endl;
  }
  if(search_stats_->auxengine_stopped_[index]){
    if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << "Thread: " << index << " Setting auxengine_stopped_ to false.";
    search_stats_->auxengine_stopped_[index] = false;    
  }
  search_stats_->auxengine_stopped_mutex_.unlock();

  std::string prev_line;
  std::string my_line;
  std::string line;
  std::string token;
  std::string my_token;  
  bool stopping = false;
  bool second_stopping = false;
  bool third_stopping = false;
  bool second_stopping_notification = false;
  while(std::getline(*search_stats_->vector_of_ipstreams[index], line)) {
    if (params_.GetAuxEngineVerbosity() >= 9 &&
	!second_stopping_notification) {
      LOGFILE << "Thread: " << index << " auxe:" << line;
    }

    std::istringstream iss(line);
    iss >> token >> std::ws;

    if (token == "bestmove") {
      iss >> token;
      if(token == "info"){
	if (params_.GetAuxEngineVerbosity() >= 1) LOGFILE << "Hit a case of https://github.com/hans-ekbrand/lc0/issues/9";
	// This is a case of https://github.com/hans-ekbrand/lc0/issues/9
	// bestmove:info" indicates something is corrupted in the input stream.
	// issue `stop`, stay in the loop and try another iteration.
	// TODO: If the next iteration also fails, stop and restart the engine.
	search_stats_->auxengine_stopped_mutex_.lock();
	*search_stats_->vector_of_opstreams[index] << "stop" << std::endl;
	search_stats_->auxengine_stopped_mutex_.lock();	
      } else {
	break;
      }
    }
    prev_line = line;

    // Don't send a second stop command
    if (!stopping) {
      stopping = stop_.load(std::memory_order_acquire);
      if (stopping) {
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine(), Thread=" << index << " caught a stop signal 2.";	
        // (unless someone else already has sent stop) send stop,
	// stay in loop to get best response, otherwise it
        // will disturb the next iteration.
	// only send stop if we are the first to detect that search has stopped.
	search_stats_->auxengine_stopped_mutex_.lock();
	if(!search_stats_->auxengine_stopped_[index]){
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine(), Thread=" << index << " Stopping the A/B helper Start";
	  *search_stats_->vector_of_opstreams[index] << "stop" << std::endl; // stop the A/B helper	  
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine(), Thread=" << index << " Stopping the A/B helper Stop";
	  search_stats_->auxengine_stopped_[index] = true;
	} else {
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "MaybeTriggerStop() must have already sent stop to the engine for instance." << index;
	}
	search_stats_->auxengine_stopped_mutex_.unlock();
      } else {
	// Since we are not stopping, do the ordinary stuff
	// parse and queue PV:s even before the search is finished, if the depth is high enough (which will be determined by AuxEncode_and_Enqueue().
	// but only use this if this is indefinite exploration, otherwise we just get a lot of junk.	
	if (token == "info" && infinite_exploration) {
	  // Since we (possibly) now create multiple PV:s per node, also (possibly) add a source.
	  // search_stats_->auxengine_mutex_.lock(); // take a lock even if we are just reading
	  // int source = search_stats_->source_of_queued_nodes.front();
	  // search_stats_->auxengine_mutex_.unlock();
	  int source = 1; // dummy
	  AuxEncode_and_Enqueue(line, depth, my_board, my_position, my_moves_from_the_white_side, source, true, index);
	}
      }
    } else {
      // Stopping is true, but did it happen before or after the helper sent its info line? Assume it was after, in which case the helper is all good.
      if(second_stopping){
	// inspecting the output from the helpers, suggest that it is harmless, just a normal info pv line.
	// perhaps they output at least one such line, and bestmove will come next?
	search_stats_->auxengine_stopped_mutex_.lock();
	*search_stats_->vector_of_opstreams[index] << "stop" << std::endl; // stop the A/B helper
	search_stats_->auxengine_stopped_mutex_.unlock();
	if (third_stopping && params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Thread: " << index << " We found that search was stopped on the previous iteration, but the current line from the helper was not 'bestmove'. Probably the helper engine does not repond to stop until it has search for some minimum amount of time (like 10 ms). As a workaround send yet another stop. This is the output from the helper: " << line;
	if(!third_stopping){
	  third_stopping = true;
	}
      } else {
        second_stopping = true;
      }
    }
  }
  if (stopping) {
    // Don't use results of a search that was stopped.
    // Not because the are unreliable, but simply because we want to shut down as fast as possible.
    return;
  }
  search_stats_->auxengine_stopped_mutex_.lock();
  search_stats_->auxengine_stopped_[index] = true; // stopped means "not running". It does not mean it was stopped prematurely.
  search_stats_->auxengine_stopped_mutex_.unlock();
  
  if (params_.GetAuxEngineVerbosity() >= 9) {
    LOGFILE << "Thread: " << index << " pv:" << prev_line;
    LOGFILE << "Thread: " << index << " bestanswer:" << token;
  }
  if(prev_line == ""){
    if (params_.GetAuxEngineVerbosity() >= 1) LOGFILE << "Thread: " << index << " Empty PV, returning early from doAuxEngine().";
    // TODO restart the helper engine?
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
    return;
  }
  if (! search_stats_->vector_of_children[index]->running()) {
    LOGFILE << "Thread: " << index << " AuxEngine died!";
    throw Exception("AuxEngine died!");
  }
  auto auxengine_dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - auxengine_start_time)
      .count();
  auxengine_total_dur += auxengine_dur;
  auxengine_num_evals++;

  // search_stats_->auxengine_mutex_.lock();
  // int source = search_stats_->source_of_queued_nodes.front();
  // search_stats_->source_of_queued_nodes.pop();
  // search_stats_->auxengine_mutex_.unlock();
  int source = 1; // dummy
  
  AuxEncode_and_Enqueue(prev_line, depth, my_board, my_position, my_moves_from_the_white_side, source, false, index);  

}

void Search::AuxWait() {
  if (params_.GetAuxEngineVerbosity() >= 7) LOGFILE << "In AuxWait()";
  while (!auxengine_threads_.empty()) {
    Mutex::Lock lock(threads_mutex_);
    auxengine_threads_.back().join();
    auxengine_threads_.pop_back();
  }
  if (params_.GetAuxEngineVerbosity() >= 7) LOGFILE << "AuxWait finished shutting down AuxEngineWorker() threads.";

  // Clear the PV cache.
  search_stats_->my_pv_cache_mutex_.lock();
  int pv_cache_size = search_stats_->my_pv_cache_.size();
  search_stats_->my_pv_cache_.clear();
  search_stats_->my_pv_cache_mutex_.unlock();  
  LOGFILE << "5";    

  search_stats_->auxengine_mutex_.lock();

  // Store the size of the queue, for possible adjustment of threshold and time
  search_stats_->AuxEngineQueueSizeAtMoveSelectionTime = search_stats_->persistent_queue_of_nodes.size();
  search_stats_->Total_number_of_nodes = root_node_->GetN() - search_stats_->Total_number_of_nodes;
  if(params_.GetAuxEngineVerbosity() >= 4) LOGFILE << search_stats_->AuxEngineQueueSizeAtMoveSelectionTime << " nodes left in the query queue at move selection time. Threshold used: " << search_stats_->AuxEngineThreshold;

  // purge obsolete nodes in the helper queues. Note that depending on the move of the opponent more nodes can become obsolete.
  if(search_stats_->persistent_queue_of_nodes.size() > 0){

    // // just unset their status
    // long unsigned int my_size = search_stats_->persistent_queue_of_nodes.size();
    // for(long unsigned int i=0; i < my_size; i++){
    //   Node * n = search_stats_->persistent_queue_of_nodes.front(); // read the element
    //   search_stats_->persistent_queue_of_nodes.pop(); // remove it from the queue.
    //   n->SetAuxEngineMove(0xffff); // magic for pending
    // }
    
    std::queue<Node*> persistent_queue_of_nodes_temp;
    // std::queue<int> source_of_queued_nodes_temp;
    long unsigned int my_size = search_stats_->persistent_queue_of_nodes.size();
    for(long unsigned int i=0; i < my_size; i++){
      Node * n = search_stats_->persistent_queue_of_nodes.front(); // read the element
      search_stats_->persistent_queue_of_nodes.pop(); // remove it from the queue.
      // int source = search_stats_->source_of_queued_nodes.front(); // read the element
      // search_stats_->source_of_queued_nodes.pop(); // remove it from the queue.
      for (Node* n2 = n; n2 != root_node_ ; n2 = n2->GetParent()) {
	// if purge at search start never happened (because of only one move possible, auxworker() never started), then we can have disconnected nodes in the queue.
	// if(n2->GetParent() == nullptr || n2->GetParent()->GetParent() == nullptr) break;
	if(n2->GetParent() == nullptr || n2->GetParent()->GetParent() == nullptr || n2->GetParent()->GetOwnEdge() == nullptr) break;
	if(n2->GetParent()->GetParent() == root_node_){
	  if(n2->GetParent()->GetOwnEdge()->GetMove(played_history_.IsBlackToMove()) == final_bestmove_){
	    persistent_queue_of_nodes_temp.push(n);
	    // in order to be able to purge nodes that became obsolete and deallocated due to the move of the opponent,
	    // also save the grandparent that will become root at next iteration if this node is still relevant by then.
	    persistent_queue_of_nodes_temp.push(n2);
	    // source_of_queued_nodes_temp.push(source);
	  }
	  break;
	}
      }
    }
    long unsigned int size_kept = persistent_queue_of_nodes_temp.size() / 2;
    for(long unsigned int i=0; i < size_kept * 2; i++){
      search_stats_->persistent_queue_of_nodes.push(persistent_queue_of_nodes_temp.front());
      persistent_queue_of_nodes_temp.pop();
    }
      
    if(params_.GetAuxEngineVerbosity() >= 4)
      LOGFILE << "Purged " << my_size - size_kept
	      << " nodes in the query queue based the selected move: " << final_bestmove_.as_string()
	      << ". " << size_kept << " nodes remain. Sanity check size is " << search_stats_->persistent_queue_of_nodes.size();
    search_stats_->AuxEngineQueueSizeAfterPurging = size_kept;
  } else {
    if(params_.GetAuxEngineVerbosity() >= 4)      
      LOGFILE << "No nodes in the query queue at move selection";
  }

  // search_stats_->final_purge_run = true; // Inform Search::AuxEngineWorker(), which can start *AFTER* us, that we have already purged stuff. If they also do it, things will break badly.
  
  search_stats_->Number_of_nodes_added_by_AuxEngine = search_stats_->Number_of_nodes_added_by_AuxEngine + auxengine_num_updates;
  float observed_ratio = float(search_stats_->Number_of_nodes_added_by_AuxEngine) / search_stats_->Total_number_of_nodes;

  // // Decrease the EngineTime if we're in an endgame.
  // ChessBoard my_board = played_history_.Last().GetBoard();
  // if((my_board.ours() | my_board.theirs()).count() < 20){
  //   search_stats_->AuxEngineTime = std::max(10, int(std::round(params_.GetAuxEngineTime() * 0.50f))); // minimum 10 ms.
  // }

  // Time based queries    
  if (params_.GetAuxEngineVerbosity() >= 3) LOGFILE << "Summaries per move: (Time based queries) persistent_queue_of_nodes size at the end of search: " << search_stats_->AuxEngineQueueSizeAtMoveSelectionTime
	  << " Ratio added/total nodes: " << observed_ratio << " (added=" << search_stats_->Number_of_nodes_added_by_AuxEngine << "; total=" << search_stats_->Total_number_of_nodes << ")."
      << " Average duration " << (auxengine_num_evals ? (auxengine_total_dur / auxengine_num_evals) : -1.0f) << "ms"
      << " AuxEngineTime for next iteration " << search_stats_->AuxEngineTime
      << " New AuxEngineThreshold for next iteration " << search_stats_->AuxEngineThreshold
      << " Number of evals " << auxengine_num_evals
      << " Number of added nodes " << search_stats_->Number_of_nodes_added_by_AuxEngine
      << " Entries in the PV cache: " << pv_cache_size
      << " Called AuxMaybeEnqueueNode() " << number_of_times_called_AuxMaybeEnqueueNode_ << " times.";

  // Reset counters for the next move:
  LOGFILE << "1";
  search_stats_->Number_of_nodes_added_by_AuxEngine = 0;
  LOGFILE << "2";
  search_stats_->Total_number_of_nodes = 0;
  LOGFILE << "3";  
  search_stats_->auxengine_mutex_.unlock();
  LOGFILE << "4";    

  // initial_purge_run needs another lock.
  search_stats_->pure_stats_mutex_.lock();
  LOGFILE << "6";      
  search_stats_->initial_purge_run = false;
  LOGFILE << "7";      
  search_stats_->pure_stats_mutex_.unlock();
  LOGFILE << "8";      
  // Empty the other queue.
  search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock();
  LOGFILE << "9";        
  if(search_stats_->fast_track_extend_and_evaluate_queue_.empty()){
    if (params_.GetAuxEngineVerbosity() >= 4) LOGFILE << "No PVs in the fast_track_extend_and_evaluate_queue";
  } else {
    search_stats_->fast_track_extend_and_evaluate_queue_ = {};
    if (params_.GetAuxEngineVerbosity() >= 4) LOGFILE << search_stats_->fast_track_extend_and_evaluate_queue_.size() << " possibly obsolete PV:s in the queue, checking which of them are still relevant based on our move " << final_bestmove_.as_string();

    // Check if the first move in each PV is the move we played
    // Store the PVs that are still relevant in a temporary queue
    std::queue<std::vector<Move>> fast_track_extend_and_evaluate_queue_temp_;
    long unsigned int my_size = search_stats_->fast_track_extend_and_evaluate_queue_.size();
    for(long unsigned int i=0; i < my_size; i++){
      std::vector<Move> pv = search_stats_->fast_track_extend_and_evaluate_queue_.front();
      search_stats_->fast_track_extend_and_evaluate_queue_.pop();
      // final_bestmove_ is not necessarily from white's point of view.
      // but pv[0] is always from white's point of view.
      Move m;
      Move::ParseMove(&m, pv[0].as_string(), played_history_.IsBlackToMove());
      // m is now rotated if needed
      if(m == final_bestmove_){
	// remove the first move, which is the move we just played
	pv.erase(pv.begin());
	fast_track_extend_and_evaluate_queue_temp_.push(pv);
      }
    }
    // Empty the queue and copy back the relevant ones.
    search_stats_->fast_track_extend_and_evaluate_queue_ = {};
    long unsigned int size_kept = fast_track_extend_and_evaluate_queue_temp_.size();
    for(long unsigned int i=0; i < size_kept; i++){
      search_stats_->fast_track_extend_and_evaluate_queue_.push(fast_track_extend_and_evaluate_queue_temp_.front());
      fast_track_extend_and_evaluate_queue_temp_.pop();
    }
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Number of PV:s in the queue after purging: " << search_stats_->fast_track_extend_and_evaluate_queue_.size();
  }
  search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxWait done search_stats_ at: " << &search_stats_;
}

}  // namespace lczero
