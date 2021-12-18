/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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

namespace lczero {

boost::process::ipstream Search::auxengine_is_;
boost::process::opstream Search::auxengine_os_;
boost::process::child Search::auxengine_c_;
bool Search::auxengine_ready_ = false;

void Search::OpenAuxEngine() REQUIRES(threads_mutex_) {
  if (params_.GetAuxEngineFile() == "") return;
  auxengine_threads_.emplace_back([this]() { AuxEngineWorker(); });
}

void SearchWorker::AuxMaybeEnqueueNode(Node* n) {
  // the caller (DoBackupUpdate()->DoBackupUpdateSingleNode()) has a lock on search_->nodes_mutex_, so no other thread will change n right now.

  // TODO change name to AuxEnqueueNode since the choice is moved to the caller now.
  
  LOGFILE << "AuxMaybeEnqueueNode() picked node: " << n->DebugString() << " for the auxengine_queue.";

  n->SetAuxEngineMove(0xfffe); // magic for pending

  search_->auxengine_mutex_.lock();
  LOGFILE << "Size of search_->auxengine_queue_ is " << search_->auxengine_queue_.size();
  search_->auxengine_queue_.push(n);
  search_->auxengine_cv_.notify_one();
  search_->auxengine_mutex_.unlock();
}

void Search::AuxEngineWorker() {
  int number_of_pvs_delivered = 0;

  if (!auxengine_ready_) {
    auxengine_c_ = boost::process::child(params_.GetAuxEngineFile(), boost::process::std_in < auxengine_os_, boost::process::std_out > auxengine_is_);
    {
      std::istringstream iss(params_.GetAuxEngineOptions());
      std::string token;
      while(std::getline(iss, token, '=')) {
        std::ostringstream oss;
        oss << "setoption name " << token;
        std::getline(iss, token, ';');
        oss << " value " << token;
        LOGFILE << oss.str();
        auxengine_os_ << oss.str() << std::endl;
      }
      auxengine_os_ << "uci" << std::endl;
    }
    std::string line;
    while(std::getline(auxengine_is_, line)) {
      LOGFILE << line;
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
            LOGFILE << oss.str();
            auxengine_os_ << oss.str() << std::endl;
          }
        }
      }
    }
    auxengine_ready_ = true;
  }

  // Kickstart with the root node, no need to wait for it to get some
  // amount of visits. Except if root is not yet expanded, or lacks
  // edges for any other reason (e.g. being terminal), in which case we
  // should wait.
  nodes_mutex_.lock(); // write lock
  if(root_node_->GetNumEdges() > 0){
    // root is extended.
    root_node_->SetAuxEngineMove(0xfffe); // mark root as pending and queue it
    auxengine_mutex_.lock(); 
    auxengine_queue_.push(root_node_);
    auxengine_cv_.notify_one();
    auxengine_mutex_.unlock();
  }
  nodes_mutex_.unlock(); // write unlock

  Node* n;
  while (!stop_.load(std::memory_order_acquire)) {
    {
  	std::unique_lock<std::mutex> lock(auxengine_mutex_);
  	// auxengine_mutex_.lock(); 
  	// Wait until there's some work to compute.
  	auxengine_cv_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) || !auxengine_queue_.empty(); });
  	if (stop_.load(std::memory_order_acquire)) {
  	  auxengine_mutex_.unlock(); 	
  	  break;
  	}
  	n = auxengine_queue_.front();
  	auxengine_queue_.pop();
    } // release lock
    DoAuxEngine(n);
    ++number_of_pvs_delivered;
  }
  auxengine_mutex_.unlock();    
  LOGFILE << "AuxEngineWorker done, delivered " << number_of_pvs_delivered << " PVs.";
}

void Search::DoAuxEngine(Node* n) {
  // before trying to take a lock on nodes_mutex_, always check if search has stopped, in which we return early
  if(stop_.load(std::memory_order_acquire)) {
    LOGFILE << "DoAuxEngine caught a stop signal beforing doing anything.";
    return;
  }
  nodes_mutex_.lock_shared();
  LOGFILE << "DoAuxEngine() called for node" << n->DebugString();
  nodes_mutex_.unlock_shared();  
  // Calculate depth in a safe way. Return early if root cannot be
  // reached from n.
  // Todo test if this lock is unnecessary when solidtree is disabled.
  int depth = 0;
  if(n != root_node_){
    if(stop_.load(std::memory_order_acquire)) {
      LOGFILE << "DoAuxEngine caught a stop signal while calculating depth.";
      return;
    }
    nodes_mutex_.lock_shared();
    for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      depth++;
      if(n2 == nullptr){
	LOGFILE << "1. Could not reach root";
	nodes_mutex_.unlock_shared();
	return;
      } 
    }
    nodes_mutex_.unlock_shared();    
  }

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
      LOGFILE << "DoAuxEngine caught a stop signal while populating my_moves.";
      return;
    }
    nodes_mutex_.lock_shared();  
    for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      my_moves.push_back(n2->GetOwnEdge()->GetMove(flip));
      my_moves_from_the_white_side.push_back(n2->GetOwnEdge()->GetMove());
      flip = !flip;
    }
    nodes_mutex_.unlock_shared();
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

  if (params_.GetAuxEngineVerbosity() >= 1) {
    LOGFILE << "add pv=" << s << " from root position: " << GetFen(played_history_.Last());
  }
  LOGFILE << "trying to get a FEN from my_position";
  s = "position fen " + GetFen(my_position);
  LOGFILE << "got a FEN from my_position";  
  LOGFILE << "input to helper: " << s;
  
  // Before starting, test if stop_ is set
  if (stop_.load(std::memory_order_acquire)) {
    if (params_.GetAuxEngineVerbosity() >= 5) {    
      LOGFILE << "DoAuxEngine caught a stop signal 1.";
    }
    return;
  }

  auto auxengine_start_time = std::chrono::steady_clock::now();
  auxengine_os_ << s << std::endl;
  auxengine_os_ << "go depth " << params_.GetAuxEngineDepth() << std::endl;

  auxengine_stopped_mutex_.lock();
  if(auxengine_stopped_){
    auxengine_stopped_ = false;    
  }
  auxengine_stopped_mutex_.unlock();

  std::string prev_line;
  std::string line;
  std::string token;
  bool stopping = false;
  while(std::getline(auxengine_is_, line)) {
    if (params_.GetAuxEngineVerbosity() >= 2) {
      LOGFILE << "auxe:" << line;
    }
    std::istringstream iss(line);
    iss >> token >> std::ws;
    if (token == "bestmove") {
      iss >> token;
      break;
    }
    prev_line = line;

    // Don't send a second stop command
    if (!stopping) {
      stopping = stop_.load(std::memory_order_acquire);
      if (stopping) {
	if (params_.GetAuxEngineVerbosity() >= 5) {
	  LOGFILE << "DoAuxEngine caught a stop signal 2.";	
	}
        // (unless someone else already has sent stop) send stop,
	// stay in loop to get best response, otherwise it
        // will disturb the next iteration.
	// only send stop if we are the first to detect that search has stopped.
	auxengine_stopped_mutex_.lock();
	if(!auxengine_stopped_){
	  LOGFILE << "DoAuxEngine() Stopping the A/B helper Start";
	  auxengine_os_ << "stop" << std::endl; // stop the A/B helper
	  LOGFILE << "DoAuxEngine() Stopping the A/B helper Stop";
	  auxengine_stopped_ = true;
	}
	auxengine_stopped_mutex_.unlock();	
      }
    }
  }
  if (stopping) {
    // TODO: actually do use the result, if the depth achieved was the
    // requested depth and the line is actually played.

    // Don't use results of a search that was stopped.
    LOGFILE << "AUX Search was interrupted, the results will not be used";
    return;
  }
  auxengine_stopped_mutex_.lock();
  auxengine_stopped_ = true; // stopped means "not running". It does not mean it was stopped prematurely.
  auxengine_stopped_mutex_.unlock();
  
  if (params_.GetAuxEngineVerbosity() >= 1) {
    LOGFILE << "pv:" << prev_line;
    LOGFILE << "bestanswer:" << token;
  }
  if(prev_line == ""){
    LOGFILE << "Empty PV, returning early from doAuxEngine().";
    return;
  }
  if (!auxengine_c_.running()) {
    LOGFILE << "AuxEngine died!";
    throw Exception("AuxEngine died!");
  }
  auto auxengine_dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - auxengine_start_time)
      .count();
  auxengine_total_dur += auxengine_dur;
  auxengine_num_evals++;
  std::istringstream iss(prev_line);
  std::string pv;
  std::vector<uint16_t> pv_moves;

  flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);

  auto bestmove_packed_int = Move(token, !flip).as_packed_int();
  // depth is distance between root and the starting point for the auxengine
  // params_.GetAuxEngineDepth() is the depth of the requested search
  // The actual PV is often times longer, whether or not to use those extra plies is an open question.
  int pv_length = 1;
  int max_pv_length = 100;
  // int max_pv_length = depth + params_.GetAuxEngineDepth();
  // LOGFILE << "capping PV at length: " << max_pv_length << ", sum of depth = " << depth << " and AuxEngineDepth = " << params_.GetAuxEngineDepth();
  // int max_pv_length = depth + params_.GetAuxEngineFollowPvDepth();  
  // LOGFILE << "capping PV at length: " << max_pv_length << ", sum of depth = " << depth << " and AuxEngineDepth = " << params_.GetAuxEngineFollowPvDepth();  

  fast_track_extend_and_evaluate_queue_mutex_.lock(); // lock this queue before starting to modify it
  
  while(iss >> pv >> std::ws) {
    if (pv == "pv") {
      while(iss >> pv >> std::ws) {
        Move m;
        if (!Move::ParseMove(&m, pv, !flip)) {	
          if (params_.GetAuxEngineVerbosity() >= 2) {
            LOGFILE << "Ignore bad pv move: " << pv;
          }
          break;
        }
	// convert to Modern encoding, update the board and the position
	// For the conversion never flip the board. Only flip the board when you need to apply the move!
	Move m_in_modern_encoding = my_board.GetModernMove(m);

	if (my_board.flipped()) m_in_modern_encoding.Mirror();
	// Should the move applied be modern or legacy, or does it not matter?
	m_in_modern_encoding = my_board.GetModernMove(m_in_modern_encoding);
	// my_board.ApplyMove(m_in_modern_encoding); // Todo verify the correctness here, e.g. by printing a FEN.
	my_board.ApplyMove(m); // Todo verify the correctness here, e.g. by printing a FEN.	
	my_position = Position(my_position, m_in_modern_encoding);

	if (my_board.flipped()) m_in_modern_encoding.Mirror();
	my_board.Mirror();

	// my_moves.push_back(m); // Add the PV to the queue
	my_moves_from_the_white_side.push_back(m_in_modern_encoding); // Add the PV to the queue	
        pv_moves.push_back(m_in_modern_encoding.as_packed_int());
        flip = !flip;
	if(pv_length == max_pv_length) break;
	pv_length++;
      }
    }
  }

  if (pv_moves.size() == 0) {
    if (params_.GetAuxEngineVerbosity() >= 1) {
      LOGFILE << "Warning: the helper did not give a PV, will only use bestmove:" << bestmove_packed_int;
    }
    pv_moves.push_back(bestmove_packed_int);
  }
  
  std::string debug_string;
  for(int i = 0; i < (int) my_moves_from_the_white_side.size(); i++){
    debug_string = debug_string + my_moves_from_the_white_side[i].as_string() + " ";
  }
  if(played_history_.IsBlackToMove()){
    LOGFILE << "debug info: length of PV given to helper engine: " << depth << " position given to helper: " << s << " black to move at root, length of my_moves_from_the_white_side " << my_moves_from_the_white_side.size() << " my_moves_from_the_white_side: " << debug_string;
  } else {
    LOGFILE << "debug info: length of PV given to helper engine: " << depth << " position given to helper: " << s << " white to move at root, length of my_moves_from_the_white_side " << my_moves_from_the_white_side.size() << " my_moves_from_the_white_side: " << debug_string;    
  }
  
  fast_track_extend_and_evaluate_queue_.push(my_moves_from_the_white_side); // I think push() means push_back for queues.
  fast_track_extend_and_evaluate_queue_mutex_.unlock();
  LOGFILE << "Returning from DoAuxEngine()";
}

// void Search::AuxWait() REQUIRES(threads_mutex_) {
void Search::AuxWait() {  
  LOGFILE << "AuxWait start for thread: " << std::hash<std::thread::id>{}(std::this_thread::get_id());

  while (!auxengine_threads_.empty()) {
    Mutex::Lock lock(threads_mutex_);
    LOGFILE << "AuxWait about to pop threads";
    auxengine_threads_.back().join();
    auxengine_threads_.pop_back();
  }
  LOGFILE << "Summaries per move: auxengine_queue_ size at the end of search: " << auxengine_queue_.size()
      << " Average duration " << (auxengine_num_evals ? (auxengine_total_dur / auxengine_num_evals) : -1.0f) << "ms"
      << " Number of evals " << auxengine_num_evals
      << " Number of added nodes " << auxengine_num_updates;
  // TODO: For now with this simple queue method,
  // mark unfinished nodes not done again, and delete the queue.
  // Next search iteration will fill it again.

  // Assume the caller has locked nodes_mutex_
  auxengine_mutex_.lock();  
  while (!auxengine_queue_.empty()) {
    auto n = auxengine_queue_.front();
    assert(n->GetAuxEngineMove() != 0xffff); // TODO find out why this is here!
    if (n->GetAuxEngineMove() == 0xfffe) {
      n->SetAuxEngineMove(0xffff);
    }
    auxengine_queue_.pop();
  }
  
  auxengine_mutex_.unlock();
  
  // Empty the other queue.
  fast_track_extend_and_evaluate_queue_mutex_.lock();
  if(fast_track_extend_and_evaluate_queue_.empty()){
    LOGFILE << "No PVs in the fast_track_extend_and_evaluate_queue";
  } else {
    LOGFILE << fast_track_extend_and_evaluate_queue_.size() << " possibly obsolete PV:s in the queue.";
    fast_track_extend_and_evaluate_queue_ = {}; // should be faster than pop() but it is safe?
    while (!fast_track_extend_and_evaluate_queue_.empty()) {
      fast_track_extend_and_evaluate_queue_.pop();
    }
    LOGFILE << "Number of PV:s in the queue=" << fast_track_extend_and_evaluate_queue_.size();
  }
  fast_track_extend_and_evaluate_queue_mutex_.unlock();  
  LOGFILE << "AuxWait done";
}

}  // namespace lczero
