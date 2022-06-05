/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#pragma once

#include <boost/process.hpp>
#include <queue>
#include <array>
#include <condition_variable>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <algorithm>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "mcts/node.h"
#include "mcts/params.h"
#include "mcts/stoppers/timemgr.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "syzygy/syzygy.h"
#include "utils/logging.h"
#include "utils/mutex.h"
#include "utils/numa.h"

namespace lczero {

class Search {

 public:

  struct adjust_policy_stats {
    std::queue<std::vector<Node*>> queue_of_vector_of_nodes_from_helper_added_by_this_thread;
    std::queue<int> starting_depth_of_PVs_;    
    std::queue<int> amount_of_support_for_PVs_;
  };

  struct SearchStats {

    SharedMutex pure_stats_mutex_;
    Mutex fast_track_extend_and_evaluate_queue_mutex_;
    Mutex vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_ ACQUIRED_AFTER(Search::nodes_mutex_);
    MyMutex auxengine_mutex_ ACQUIRED_AFTER(Search::nodes_mutex_); // Mutex does not work with .wait().    
    Mutex auxengine_listen_mutex_;
    Mutex auxengine_stopped_mutex_ ACQUIRED_AFTER(auxengine_mutex_);
    Mutex my_pv_cache_mutex_;
    SharedMutex best_move_candidates_mutex; // For some reason this leads to a deadlock very early on. // is that comment obsolete by now?
    // Mutex best_move_candidates_mutex;
    // std::shared_mutex best_move_candidates_mutex; //fails
    // std::mutex best_move_candidates_mutex; // works
    SharedMutex test_mutex_;
    
    std::queue<Node*> persistent_queue_of_nodes GUARDED_BY(auxengine_mutex_); // the query queue for the auxillary helper engine. // clang messes thread safety analysis up since type is std::mutex
    // std::queue<Node*> persistent_queue_of_nodes; // the query queue for the auxillary helper engine.    
    // std::queue<int> source_of_queued_nodes; // 0 = SearchWorker::PickNodesToExtendTask(); 1 = Search::DoBackupUpdateSingleNode(); 2 = Search::SendUciInfo(); 3 = Search::AuxEngineWorker() node is root
    std::queue<std::vector<Move>> fast_track_extend_and_evaluate_queue_ GUARDED_BY(fast_track_extend_and_evaluate_queue_mutex_); // PV:s to be extended in Leelas search tree.
    std::queue<int> amount_of_support_for_PVs_ GUARDED_BY(fast_track_extend_and_evaluate_queue_mutex_); // Whenever an element from fast_track_extend_and_evaluate_queue_ is popped by PreExt...(), record the number of nodes to support for that PV in this vector. This way MaybeAdjustPolicyForHelperAddedNodes() can guesstimate the number of nodes there are to support an added node.
    std::queue<int> starting_depth_of_PVs_ GUARDED_BY(fast_track_extend_and_evaluate_queue_mutex_); // needed to calculate the estimated number of nodes in support for an added node.

    bool helper_thinks_it_is_better GUARDED_BY(best_move_candidates_mutex) = false;    
    bool winning_ GUARDED_BY(best_move_candidates_mutex) = false;
    bool stop_a_blunder_ GUARDED_BY(best_move_candidates_mutex) = false;
    bool save_a_win_ GUARDED_BY(best_move_candidates_mutex) = false;
    bool winning_threads_adjusted GUARDED_BY(best_move_candidates_mutex) = false;
    int non_winning_root_threads_ GUARDED_BY(best_move_candidates_mutex); // only parse once, store the result in this variable so that we can reset without parsing again.
    Move winning_move_ GUARDED_BY(best_move_candidates_mutex);
    std::vector<Move> helper_PV GUARDED_BY(best_move_candidates_mutex); // Full PV from the helper, used to find where Leela and helper diverge.
    std::vector<Move> Leelas_PV GUARDED_BY(best_move_candidates_mutex); // Full PV from PV.
    int PVs_diverge_at_depth GUARDED_BY(best_move_candidates_mutex) = 0;
    float helper_eval_of_root GUARDED_BY(best_move_candidates_mutex) = 0;
    float helper_eval_of_leelas_preferred_child GUARDED_BY(best_move_candidates_mutex) = 0;
    float helper_eval_of_helpers_preferred_child GUARDED_BY(best_move_candidates_mutex) = 0;
    int number_of_nodes_in_support_for_helper_eval_of_root GUARDED_BY(best_move_candidates_mutex) = 0;
    int number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child GUARDED_BY(best_move_candidates_mutex) = 0;

    // Node* Leelas_preferred_child_node_; // Not used currently, was used in stoppers.cc
    Node* Helpers_preferred_child_node_ GUARDED_BY(vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_); // protected by search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_
    Node* Helpers_preferred_child_node_in_Leelas_PV_ GUARDED_BY(vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_);
    std::vector<Move> vector_of_moves_from_root_to_Helpers_preferred_child_node_ GUARDED_BY(vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_);
    std::vector<Move> vector_of_moves_from_root_to_Helpers_preferred_child_node_in_Leelas_PV_ GUARDED_BY(vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_); // This is guaranteed to be of length zero unless there exists both a first and a second divergence.

    std::vector<std::shared_ptr<boost::process::ipstream>> vector_of_ipstreams; // each pointer is only used by one thread, so no protection needed
    std::vector<std::shared_ptr<boost::process::child>> vector_of_children; // each pointer is only used by one thread, so no protection needed
    std::vector<std::shared_ptr<boost::process::opstream>> vector_of_opstreams GUARDED_BY(auxengine_stopped_mutex_);
    std::vector<bool> auxengine_stopped_ GUARDED_BY(auxengine_stopped_mutex_);
    
    std::vector<bool> vector_of_auxengine_ready_; // each pointer is only used by one thread, so no protection needed
    int thread_counter GUARDED_BY(pure_stats_mutex_) = 0;

    std::map<std::string, bool> my_pv_cache_ GUARDED_BY(my_pv_cache_mutex_);

    std::condition_variable auxengine_cv_ GUARDED_BY(auxengine_mutex_);

    std::queue<Node*> nodes_added_by_the_helper; // this is useful only to assess how good the different sources are, it does not affect search
    std::queue<int> source_of_added_nodes; // 0 = SearchWorker::PickNodesToExtendTask(); 1 = Search::DoBackupUpdateSingleNode(); 2 = Search::SendUciInfo(); 3 = Search::AuxEngineWorker() node is root
    
    int AuxEngineTime GUARDED_BY(auxengine_mutex_); // dynamic version of the UCI option AuxEngineTime.
    unsigned long long int Total_number_of_nodes; // all nodes ever added to the tree.
    unsigned long long int Number_of_nodes_added_by_AuxEngine GUARDED_BY(search_stats_->pure_stats_mutex_); // all nodes ever added by the auxillary engine.
    int AuxEngineThreshold GUARDED_BY(Search::nodes_mutex_); // dynamic version of the UCI option AuxEngineThreshold. Seldom written to but often read by a function that has a read-only lock on nodes, which is why it is efficient to use that mutex for it.
    int AuxEngineQueueSizeAtMoveSelectionTime;
    long unsigned int AuxEngineQueueSizeAfterPurging;
    Move ponder_move; // the move predicted by search().
    float q; // the expected q based on the predicted move.
    bool New_Game = false; // used by EngineController::NewGame in engine.cc to inform search that a new game has started, so it can re-initiate AuxEngineTime to the value given by UCI
    int size_of_queue_at_start; // used by Search::AuxEngineWorker() to decide how many node to check for purging at the start of each move. Without this, new nodes added by before the purge happened would cause a crash.
    int current_depth = 1;
    bool initial_purge_run GUARDED_BY(search_stats_->pure_stats_mutex_) = false; // used by AuxEngineWorker() thread 0 to inform subsequent threads that they should return immediately.
    std::queue<Move*> temporary_queue_of_moves; //

    // temporary stuff, but keeping them here to help clang with the thread safety analysis, but that failed anyway because auxengine_mutex_ is used with wait() which requires std::lock.
    int64_t number_of_times_called_AuxMaybeEnqueueNode_ GUARDED_BY(auxengine_mutex_) = 0 ;

  };

  Search(const NodeTree& tree, Network* network,
         std::unique_ptr<UciResponder> uci_responder,
         const MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<SearchStopper> stopper, bool infinite,
         const OptionsDict& options, NNCache* cache,
         SyzygyTablebase* syzygy_tb,
	 std::queue<Node*>* persistent_queue_of_nodes,
	 const std::shared_ptr<SearchStats> search_stats
	 );

  ~Search();

  // Starts worker threads and returns immediately.
  void StartThreads(size_t how_many);

  // Starts search with k threads and wait until it finishes.
  void RunBlocking(size_t threads);

  // Stops search. At the end bestmove will be returned. The function is not
  // blocking, so it returns before search is actually done.
  void Stop();
  // Stops search, but does not return bestmove. The function is not blocking.
  void Abort();
  // Blocks until all worker thread finish.
  void Wait();
  // Returns whether search is active. Workers check that to see whether another
  // search iteration is needed.
  bool IsSearchActive() const;

  // Returns best move, from the point of view of white player. And also ponder.
  // May or may not use temperature, according to the settings.
  std::pair<Move, Move> GetBestMove();

  // Returns the evaluation of the best move, WITHOUT temperature. This differs
  // from the above function; with temperature enabled, these two functions may
  // return results from different possible moves. If @move and @is_terminal are
  // not nullptr they are set to the best move and whether it leads to a
  // terminal node respectively.
  Eval GetBestEval(Move* move = nullptr, bool* is_terminal = nullptr) const;
  // Returns the total number of playouts in the search.
  std::int64_t GetTotalPlayouts() const;
  // Returns the search parameters.
  const SearchParams& GetParams() const { return params_; }

  // If called after GetBestMove, another call to GetBestMove will have results
  // from temperature having been applied again.
  void ResetBestMove();

  // Returns NN eval for a given node from cache, if that node is cached.
  NNCacheLock GetCachedNNEval(const Node* node) const;

  //CurrentPosition current_position_;
  std::string current_position_fen_;
  std::vector<std::string> current_position_moves_;
  std::string current_uci_;

 private:
  // Computes the best move, maybe with temperature (according to the settings).
  void EnsureBestMoveKnown();

  // Returns a child with most visits, with or without temperature.
  // NoTemperature is safe to use on non-extended nodes, while WithTemperature
  // accepts only nodes with at least 1 visited child.
  EdgeAndNode GetBestChildNoTemperature(Node* parent, int depth) const;
  std::vector<EdgeAndNode> GetBestChildrenNoTemperature(Node* parent, int count,
                                                        int depth) const;
  EdgeAndNode GetBestRootChildWithTemperature(float temperature) const;

  int64_t GetTimeSinceStart() const;
  int64_t GetTimeSinceFirstBatch() const;
  void MaybeTriggerStop(const IterationStats& stats, StoppersHints* hints);
  void MaybeOutputInfo();
  void SendUciInfo();  // Requires nodes_mutex_ to be held.
  // Sets stop to true and notifies watchdog thread.
  void FireStopInternal();

  void SendMovesStats() const;
  // Function which runs in a separate thread and watches for time and
  // uci `stop` command;
  void WatchdogThread();

  // Fills IterationStats with global (rather than per-thread) portion of search
  // statistics. Currently all stats there (in IterationStats) are global
  // though.
  void PopulateCommonIterationStats(IterationStats* stats);

  // Returns verbose information about given node, as vector of strings.
  // Node can only be root or ponder (depth 1).
  std::vector<std::string> GetVerboseStats(Node* node) const;

  // Returns the draw score at the root of the search. At odd depth pass true to
  // the value of @is_odd_depth to change the sign of the draw score.
  // Depth of a root node is 0 (even number).
  float GetDrawScore(bool is_odd_depth) const;

  // Ensure that all shared collisions are cancelled and clear them out.
  void CancelSharedCollisions();

  mutable Mutex counters_mutex_ ACQUIRED_AFTER(nodes_mutex_);
  // Tells all threads to stop.
  std::atomic<bool> stop_{false};
  // Condition variable used to watch stop_ variable.
  std::condition_variable watchdog_cv_;
  // Tells whether it's ok to respond bestmove when limits are reached.
  // If false (e.g. during ponder or `go infinite`) the search stops but nothing
  // is responded until `stop` uci command.
  bool ok_to_respond_bestmove_ GUARDED_BY(counters_mutex_) = true;
  // There is already one thread that responded bestmove, other threads
  // should not do that.
  bool bestmove_is_sent_ GUARDED_BY(counters_mutex_) = false;
  // Stored so that in the case of non-zero temperature GetBestMove() returns
  // consistent results.
  Move final_bestmove_ GUARDED_BY(counters_mutex_);
  Move final_pondermove_ GUARDED_BY(counters_mutex_);
  std::unique_ptr<SearchStopper> stopper_ GUARDED_BY(counters_mutex_);

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node* root_node_;
  NNCache* cache_;
  SyzygyTablebase* syzygy_tb_;

  // Fixed positions which happened before the search.
  const PositionHistory& played_history_;

  Network* const network_;
  const SearchParams params_;
  const MoveList searchmoves_;
  const std::chrono::steady_clock::time_point start_time_;
  std::queue<Node*>* persistent_queue_of_nodes_;
  const std::shared_ptr<SearchStats> search_stats_;
  // const std::shared_ptr<boost::process::ipstream> my_ip_ptr_;
  // std::shared_ptr<boost::process::opstream> my_op_ptr_;
  // const std::shared_ptr<boost::process::child> my_children_ptr_;      
  int64_t initial_visits_;
  // root_is_in_dtz_ must be initialized before root_move_filter_.
  bool root_is_in_dtz_ = false;
  // tb_hits_ must be initialized before root_move_filter_.
  std::atomic<int> tb_hits_{0};
  const MoveList root_move_filter_;

  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_ GUARDED_BY(nodes_mutex_);
  Edge* last_outputted_info_edge_ GUARDED_BY(nodes_mutex_) = nullptr;
  ThinkingInfo last_outputted_uci_info_ GUARDED_BY(nodes_mutex_);
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_batches_ GUARDED_BY(nodes_mutex_) = 0;
  // Maximum search depth = length of longest path taken in PickNodetoExtend.
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  // Cumulative depth of all paths taken in PickNodetoExtend.
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;

  std::optional<std::chrono::steady_clock::time_point> nps_start_time_
      GUARDED_BY(counters_mutex_);

  std::atomic<int> pending_searchers_{0};
  std::atomic<int> backend_waiting_counter_{0};
  std::atomic<int> thread_count_{0};

  std::vector<std::pair<Node*, int>> shared_collisions_
      GUARDED_BY(nodes_mutex_);

  std::unique_ptr<UciResponder> uci_responder_;
 
  void OpenAuxEngine();
  void AuxEngineWorker();
  void AuxWait();
  void DoAuxEngine(Node* n, int index);
  void AuxEncode_and_Enqueue(std::string pv_as_string, int depth, ChessBoard my_board, Position my_position, std::vector<lczero::Move> my_moves_from_the_white_side, bool require_some_depth, int thread);
  void AuxUpdateP(Node* n, std::vector<uint16_t> pv_moves, int ply, ChessBoard my_board);

  std::vector<std::thread> auxengine_threads_;
  int64_t auxengine_total_dur = 0;
  int64_t auxengine_num_evals = 0;
  int64_t auxengine_num_updates = 0;
  // when stop_ is issued, only send "Stop" via UCI to once, either from MaybeTriggerStop() or from DoAuxNode(). Once for every thread.
  std::vector<bool> auxengine_stopped_;

  friend class SearchWorker;
};

// Single thread worker of the search engine.
// That used to be just a function Search::Worker(), but to parallelize it
// within one thread, have to split into stages.
class SearchWorker {
  
 public:
  SearchWorker(Search* search, const SearchParams& params, int id)
      : search_(search),
        history_(search_->played_history_),
        params_(params),
        moves_left_support_(search_->network_->GetCapabilities().moves_left !=
                            pblczero::NetworkFormat::MOVES_LEFT_NONE) {
    Numa::BindThread(id);
    for (int i = 0; i < params.GetTaskWorkersPerSearchWorker(); i++) {
      task_workspaces_.emplace_back();
      task_threads_.emplace_back([this, i]() {
        Numa::BindThread(i);
        this->RunTasks(i);
      });
    }
  }

  ~SearchWorker() {
    {
      task_count_.store(-1, std::memory_order_release);
      Mutex::Lock lock(picking_tasks_mutex_);
      exiting_ = true;
      task_added_.notify_all();
    }
    for (size_t i = 0; i < task_threads_.size(); i++) {
      task_threads_[i].join();
    }
  }

  // Runs iterations while needed.
  void RunBlocking() {
    try {
      // A very early stop may arrive before this point, so the test is at the
      // end to ensure at least one iteration runs before exiting.
      do {
        ExecuteOneIteration();
      } while (search_->IsSearchActive());
    } catch (std::exception& e) {
      std::cerr << "Unhandled exception in worker thread: " << e.what()
                << std::endl;
      abort();
    }
  }

  // Does one full iteration of MCTS search:
  // 1. Initialize internal structures.
  // 2. Gather minibatch.
  // 3. Prefetch into cache.
  // 4. Run NN computation.
  // 5. Retrieve NN computations (and terminal values) into nodes.
  // 6. Propagate the new nodes' information to all their parents in the tree.
  // 7. Update the Search's status and progress information.
  void ExecuteOneIteration();

  // The same operations one by one:
  // 1. Initialize internal structures.
  // @computation is the computation to use on this iteration.
  void InitializeIteration(std::unique_ptr<NetworkComputation> computation);

  // 1.5 Extend tree with nodes using PV of a/b helper, and add the new
  // nodes to the minibatch
  const std::shared_ptr<Search::adjust_policy_stats> PreExtendTreeAndFastTrackForNNEvaluation();
  // std::queue<std::vector<Node*>> PreExtendTreeAndFastTrackForNNEvaluation();
  void PreExtendTreeAndFastTrackForNNEvaluation_inner(Node * my_node, std::vector<lczero::Move> my_moves, int ply, int nodes_added, int source, std::vector<Node*>* nodes_from_helper_added_by_this_PV);
  // void PreExtendTreeAndFastTrackForNNEvaluation_inner(Node * my_node,
  //     std::vector<lczero::Move> my_moves, int ply, int nodes_added, int source);
  
  // 2. Gather minibatch.
  void GatherMinibatch();
  // Variant for multigather path.
  void GatherMinibatch2(int number_of_nodes_already_added);

  // 2b. Copy collisions into shared_collisions_.
  void CollectCollisions();

  // 3. Prefetch into cache.
  void MaybePrefetchIntoCache();

  // 4. Run NN computation.
  void RunNNComputation();

  // 5. Retrieve NN computations (and terminal values) into nodes.
  void FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  void DoBackupUpdate();

  // 6.5 Check policy and V for any new nodes added by the helper, and adjust policy if V is promising
  void MaybeAdjustPolicyForHelperAddedNodes(const std::shared_ptr<Search::adjust_policy_stats> foo);

  // 7. Update the Search's status and progress information.
  void UpdateCounters();

 private:
  struct NodeToProcess {
    // bool IsExtendable() const { return !is_collision && !node->IsTerminal() && !node->HasChildren(); }
    bool IsExtendable() const { return !is_collision && !node->IsTerminal(); }    
    bool IsCollision() const { return is_collision; }
    bool CanEvalOutOfOrder() const {
      return is_cache_hit || node->IsTerminal();
    }

    // The node to extend.
    Node* node;
    // Value from NN's value head, or -1/0/1 for terminal nodes.
    float v;
    // Draw probability for NN's with WDL value head.
    float d;
    // Estimated remaining plies left.
    float m;
    int multivisit = 0;
    // If greater than multivisit, and other parameters don't imply a lower
    // limit, multivist could be increased to this value without additional
    // change in outcome of next selection.
    int maxvisit = 0;
    uint16_t depth;
    bool nn_queried = false;
    bool is_cache_hit = false;
    bool is_collision = false;
    int probability_transform = 0;

    // Details only populated in the multigather path.

    // Only populated for visits,
    std::vector<Move> moves_to_visit;

    // Details that are filled in as we go.
    uint64_t hash;
    NNCacheLock lock;
    std::vector<uint16_t> probabilities_to_cache;
    InputPlanes input_planes;
    mutable int last_idx = 0;
    bool ooo_completed = false;

    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count) {
      return NodeToProcess(node, depth, true, collision_count, 0);
    }
    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count, int max_count) {
      return NodeToProcess(node, depth, true, collision_count, max_count);
    }
    static NodeToProcess Visit(Node* node, uint16_t depth) {
      return NodeToProcess(node, depth, false, 1, 0);
    }

    // Methods to allow NodeToProcess to conform as a 'Computation'. Only safe
    // to call if is_cache_hit is true in the multigather path.

    float GetQVal(int) const { return lock->q; }

    float GetDVal(int) const { return lock->d; }

    float GetMVal(int) const { return lock->m; }

    float GetPVal(int, int move_id) const {
      const auto& moves = lock->p;

      int total_count = 0;
      while (total_count < moves.size()) {
        // Optimization: usually moves are stored in the same order as queried.
        const auto& move = moves[last_idx++];
        if (last_idx == moves.size()) last_idx = 0;
        if (move.first == move_id) return move.second;
        ++total_count;
      }
      assert(false);  // Move not found.
      return 0;
    }

   private:
    NodeToProcess(Node* node, uint16_t depth, bool is_collision, int multivisit,
                  int max_count)
        : node(node),
          multivisit(multivisit),
          maxvisit(max_count),
          depth(depth),
          is_collision(is_collision) {}
  };

  // Holds per task worker scratch data
  struct TaskWorkspace {
    std::array<Node::Iterator, 256> cur_iters;
    std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer;
    std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform;
    std::vector<int> vtp_last_filled;
    std::vector<int> current_path;
    std::vector<Move> moves_to_path;
    PositionHistory history;
    TaskWorkspace() {
      vtp_buffer.reserve(30);
      visits_to_perform.reserve(30);
      vtp_last_filled.reserve(30);
      current_path.reserve(30);
      moves_to_path.reserve(30);
      history.Reserve(30);
    }
  };

  struct PickTask {
    enum PickTaskType { kGathering, kProcessing };
    PickTaskType task_type;

    // For task type gathering.
    Node* start;
    int base_depth;
    int collision_limit;
    std::vector<Move> moves_to_base;
    std::vector<NodeToProcess> results;

    // Task type post gather processing.
    int start_idx;
    int end_idx;

    bool complete = false;

    PickTask(Node* node, uint16_t depth, const std::vector<Move>& base_moves,
             int collision_limit)
        : task_type(kGathering),
          start(node),
          base_depth(depth),
          collision_limit(collision_limit),
          moves_to_base(base_moves) {}
    PickTask(int start_idx, int end_idx)
        : task_type(kProcessing), start_idx(start_idx), end_idx(end_idx) {}
  };

  NodeToProcess PickNodeToExtend(int collision_limit);
  void ExtendNode(Node* node, int depth);
  bool AddNodeToComputation(Node* node, bool add_if_cached, int* transform_out);
  int PrefetchIntoCache(Node* node, int budget, bool is_odd_depth);
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process);
  // Returns whether a node's bounds were set based on its children.
  bool MaybeSetBounds(Node* p, float m, int* n_to_fix, float* v_delta,
                      float* d_delta, float* m_delta) const;
  void PickNodesToExtend(int collision_limit, bool override_cpuct);
  bool PickNodesToExtendTask(Node* starting_point, int collision_limit,
                             int base_depth,
                             const std::vector<Move>& moves_to_base,
                             std::vector<NodeToProcess>* receiver,
                             TaskWorkspace* workspace,
			     bool override_cpuct);
  void EnsureNodeTwoFoldCorrectForDepth(Node* node, int depth);
  void ProcessPickedTask(int batch_start, int batch_end,
                         TaskWorkspace* workspace);
  void ExtendNode(Node* node, int depth, const std::vector<Move>& moves_to_add,
                  PositionHistory* history);
  template <typename Computation>
  void FetchSingleNodeResult(NodeToProcess* node_to_process,
                             const Computation& computation,
                             int idx_in_computation);
  void RunTasks(int tid);
  void ResetTasks();
  // Returns how many tasks there were.
  int WaitForTasks();

  Search* const search_;
  // List of nodes to process.
  std::vector<NodeToProcess> minibatch_;
  std::unique_ptr<CachingComputation> computation_;
  // History is reset and extended by PickNodeToExtend().
  PositionHistory history_;
  int number_out_of_order_ = 0;
  const SearchParams& params_;
  std::unique_ptr<Node> precached_node_;
  const bool moves_left_support_;
  IterationStats iteration_stats_;
  StoppersHints latest_time_manager_hints_;

  // Multigather task related fields.

  Mutex picking_tasks_mutex_;
  std::vector<PickTask> picking_tasks_;
  std::atomic<int> task_count_ = -1;
  std::atomic<int> task_taking_started_ = 0;
  std::atomic<int> tasks_taken_ = 0;
  std::atomic<int> completed_tasks_ = 0;
  std::condition_variable task_added_;
  std::vector<std::thread> task_threads_;
  std::vector<TaskWorkspace> task_workspaces_;
  TaskWorkspace main_workspace_;
  bool exiting_ = false;

  void AuxMaybeEnqueueNode(Node* n) REQUIRES(Search::nodes_mutex_);

};

}  // namespace lczero
