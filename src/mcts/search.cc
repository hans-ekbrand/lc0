/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2022 The LCZero Authors

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
#include <array>
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

namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;
unsigned long long int number_of_skipped_playouts = 0; // Used to calculate the beta_prior in move selection
signed int this_edge_has_higher_expected_q_than_the_most_visited_child = -1; // explore this child of root more than PUCT would have done if a smart-pruning was rejected because this child was promising.
  

MoveList MakeRootMoveFilter(const MoveList& searchmoves,
                            SyzygyTablebase* syzygy_tb,
                            const PositionHistory& history, bool fast_play,
                            std::atomic<int>* tb_hits, bool* dtz_success) {
  assert(tb_hits);
  assert(dtz_success);
  // Search moves overrides tablebase.
  if (!searchmoves.empty()) return searchmoves;
  const auto& board = history.Last().GetBoard();
  MoveList root_moves;
  if (!syzygy_tb || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb->max_cardinality()) {
    return root_moves;
  }
  if (syzygy_tb->root_probe(
          history.Last(), fast_play || history.DidRepeatSinceLastZeroingMove(),
          &root_moves)) {
    *dtz_success = true;
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  } else if (syzygy_tb->root_probe_wdl(history.Last(), &root_moves)) {
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  }
  return root_moves;
}

class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}

  MEvaluator(const SearchParams& params, const Node* parent = nullptr)
      : enabled_{true},
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f},
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}

  void SetParent(const Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM();
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }

  float GetM(const EdgeAndNode& child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    const float child_m = child.GetM(parent_m_);
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  float GetM(Node* child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  // The M utility to use for unvisited nodes.
  float GetDefaultM() const { return 0.0f; }

 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
    return std::abs(parent->GetQ(0.0f)) > q_threshold;
  }

  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

}  // namespace

Search::Search(const NodeTree& tree, Network* network,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb,
	       std::queue<Node*>* persistent_queue_of_nodes,
	       std::shared_ptr<SearchStats> search_stats
	       )
    : ok_to_respond_bestmove_(!infinite),
      stopper_(std::move(stopper)),
      root_node_(tree.GetCurrentHead()),
      cache_(cache),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      params_(options),
      searchmoves_(searchmoves),
      start_time_(start_time),
      persistent_queue_of_nodes_(persistent_queue_of_nodes),
      search_stats_(search_stats),
      initial_visits_(root_node_->GetN()),
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)) {
  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
  search_stats_->best_move_candidates_mutex.lock();
  search_stats_->Leelas_PV = {};
  search_stats_->helper_PV = {};
  search_stats_->helper_PV_from_instance_two_explore_moves = {};
  search_stats_->helper_PV_from_instance_one_explore_moves = {};    
  search_stats_->vector_of_moves_from_root_to_first_minimax_divergence = {};
  search_stats_->PVs_diverge_at_depth = 0;
  search_stats_->number_of_nodes_in_support_for_helper_eval_of_root = 0;
  search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child = 0;
  search_stats_->helper_eval_of_root = 0;
  search_stats_->helper_eval_of_leelas_preferred_child = 0;
  search_stats_->helper_eval_of_helpers_preferred_child = 0;
  search_stats_->thread_one_and_two_have_started = false;
  search_stats_->best_move_candidates_mutex.unlock();
  search_stats_->auxengine_mutex_.lock();
  search_stats_->size_of_queue_at_start = search_stats_->persistent_queue_of_nodes.size();
  search_stats_->final_purge_run = false;
  search_stats_->thread_counter = 0;
  search_stats_->Number_of_nodes_added_by_AuxEngine = 0;
  search_stats_->Total_number_of_nodes = root_node_->GetN();
  if (search_stats_->AuxEngineThreshold == 0 &&
      params_.GetAuxEngineInstances() > 1){
    search_stats_->AuxEngineThreshold = params_.GetAuxEngineThreshold();
  }
  if (params_.GetAuxEngineVerbosity() >= 3) LOGFILE
       << "Search called with search_stats at: " << &search_stats_
       << " size of persistent_queue: " << search_stats_->persistent_queue_of_nodes.size()
       << " size of search tree at start: " << search_stats_->Total_number_of_nodes
       << " threshold=" << search_stats_->AuxEngineThreshold;
  search_stats_->auxengine_mutex_.unlock();
  search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock();
  search_stats_->Number_of_nodes_fast_tracked_because_of_fluctuating_eval = 0;
  search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();
}

namespace {
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}
}  // namespace

void Search::SendUciInfo(bool only_check_if_Leelas_PV_was_changed) REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  const auto max_pv = params_.GetMultiPv();
  const auto edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto display_cache_usage = params_.GetDisplayCacheUsage();
  const auto draw_score = GetDrawScore(false);

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  if (!per_pv_counters) {
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (display_cache_usage) {
    common_info.hashfull =
        cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  }
  if (nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      common_info.nps = total_playouts_ * 1000 / time_since_first_batch_ms;
    }
  }
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  const auto default_q = -root_node_->GetQ(-draw_score);
  const auto default_wl = -root_node_->GetWL();
  const auto default_d = root_node_->GetD();
  bool need_to_restart_thread_one = false;
  bool need_to_restart_thread_two = false;

  // Check if the relevant part of Leelas PV has changed
  // Take the lock once and make local copies outside the loop over edges below.
  search_stats_->best_move_candidates_mutex.lock_shared();
  std::vector<Move> local_copy_of_leelas_PV = search_stats_->Leelas_PV;
  std::vector<Move> local_copy_of_helper_PV = search_stats_->helper_PV;  
  long unsigned int local_copy_of_PVs_diverge_at_depth = search_stats_->PVs_diverge_at_depth;
  search_stats_->best_move_candidates_mutex.unlock_shared();
  std::vector<Move> local_copy_of_leelas_new_PV;
  long unsigned int depth = 0;
  bool notified_already = false;

  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    const auto wl = edge.GetWL(default_wl);
    const auto floatD = edge.GetD(default_d);
    const auto q = edge.GetQ(default_q, draw_score);
    // LOGFILE << "multipv: " << multipv << " q: " << q << " n: " << edge.GetN();
    if (edge.IsTerminal() && wl != 0.0f) {
      uci_info.mate = std::copysign(
          std::round(edge.GetM(0.0f)) / 2 + (edge.IsTbTerminal() ? 101 : 1),
          wl);
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q);
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    } else if (score_type == "centipawn_2019") {
      uci_info.score = 295 * wl / (1 - 0.976953126 * std::pow(wl, 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q * 10000;
    } else if (score_type == "W-L") {
      uci_info.score = wl * 10000;
    }

    auto w =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - floatD))));
    auto l =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - floatD))));
    // Using 1000-w-l so that W+D+L add up to 1000.0.
    auto d = 1000 - w - l;
    if (d < 0) {
      w = std::min(1000, std::max(0, w + d / 2));
      l = 1000 - w;
      d = 0;
    }
    uci_info.wdl = ThinkingInfo::WDL{w, d, l};
    if (network_->GetCapabilities().has_mlh()) {
      uci_info.moves_left = static_cast<int>(
          (1.0f + edge.GetM(1.0f + root_node_->GetM())) / 2.0f);
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = played_history_.IsBlackToMove();
    // int depth = 0;
    // bool notified_already = false;
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;  // Last edge was dangling, cannot continue.

      // Thread one must be restarted if Leelas PV changed at a depth different than local_copy_of_PVs_diverge_at_depth
      // Thread two must be restarted if Leelas PV changed at a depth lower than local_copy_of_vector_of_moves_from_root_to_Helpers_preferred_child_node_in_Leelas_PV_.size()
      // Optimise for the case of same depth: What if the change is at the same depth? If Leela now agrees with the helper => restart, if Leela still disagrees => do not restart
      
      // If there is a change, this change can result in the node of divergence is changed to a node closer to root, changed to another node at the same distance from root, changed to a node further away from root.
      // If Leelas PV was A B C D and is now A B E F and the helpers PV is A B G H, then the distance is the same, and only thread 2 needs to be restarted.
      
      // If Leelas new PV instead is A B G I, then we also need to restart helper thread 1, distance from root is now higher than before.
      // We can stop test for equal moves when we have encountered the first divergence.
      local_copy_of_leelas_new_PV.push_back(iter.GetMove()); // stored conditionally mirrored, so no flip here
      
      if (!stop_.load(std::memory_order_acquire) && // search is not stopped
	  multipv == 1 && // prefered PV
	  !notified_already && // so far the PV:s are identical
	  !need_to_restart_thread_two && // if thread two is restarted, then either depth is too high for thread 1, or thread 1 is also already restarted,
                                         // in either case we can not detect that thread one should be restarted if it is not already.
	  params_.GetAuxEngineFile() != "" && // helper is activated
	  local_copy_of_leelas_PV.size() > 0 && // There is already a PV
	  local_copy_of_leelas_PV.size() > depth && // The old PV still has moves in it that we can compare with the current PV
	  ! iter.node()->IsTerminal()){ // child is not terminal // why is that relevant? Is it because we don't want to start the helper on a terminal node?
	if(iter.GetMove().as_string() != local_copy_of_leelas_PV[depth].as_string()){
	  notified_already = true; // only check until this is true, and thus only act once.
	  if(depth < local_copy_of_PVs_diverge_at_depth){ // Disagreement is now earlier than it was before, both threads need to be restarted.
	    need_to_restart_thread_one = true;
	    need_to_restart_thread_two = true;
	    Move m;
	    Move::ParseMove(&m, local_copy_of_leelas_PV[depth].as_string(), flip);
	    if (params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "Found a relevant change in Leelas PV at depth " << depth << ". Old divergence happened at depth=" << local_copy_of_PVs_diverge_at_depth << " New divergence at depth=" << depth << " Leelas new move here is: " << iter.GetMove(flip).as_string() << " and is different from Leelas old move: " << m.as_string() << ", will thus restart both thread 1 and thread 2.";
	    // local_copy_of_PVs_diverge_at_new_depth = depth;
	  }
	  // Change in Leelas PV at the same node as the previous divergence , necessarily restart thread one, but only restart thread two if there is still a divergence.
	  // if(int(local_copy_of_vector_of_moves_from_root_to_Helpers_preferred_child_node_in_Leelas_PV_.size()) == depth){ // Why not use the same terms in the condition as in the test above?
	  if(depth == local_copy_of_PVs_diverge_at_depth && local_copy_of_helper_PV.size() > 0){ // The last condition is needed if there is no helper PV yet.
	    need_to_restart_thread_one = true;
	    // local_copy_of_PVs_diverge_at_new_depth = local_copy_of_PVs_diverge_at_depth;
	    if(iter.GetMove().as_string() == local_copy_of_helper_PV[depth].as_string()){
	      // Leela has changed her mind and does now agree with the helper. Thread two will have to find another starting point.
	      need_to_restart_thread_two = true;
	      if (params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "Leela has changed her mind and does now agree with the helper that move " << iter.GetMove(flip).as_string() << " is better than her old preference: " << local_copy_of_leelas_PV[depth].as_string() << ". will thus restarting both thread 1 and thread 2 now.";
	    } else {
	      if (params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "Leela changed her mind exactly where she and helper disagreed, but she still disagrees and prefers " << iter.GetMove(flip).as_string() << " instead of " << local_copy_of_helper_PV[depth].as_string() << " helper (thread 2) can just continue.";
	    }
	  }
	} else {
	  // They still recommend the same move
	  if(depth < local_copy_of_helper_PV.size()){
	    depth += 1;
	  }
	}
      }
    }
  }

  // Even if the threads does not need to be restarted, update Leelas PV it is has changed.
  bool leelas_pv_has_changed = false;
  if(local_copy_of_leelas_PV.size() != local_copy_of_leelas_new_PV.size()){
    leelas_pv_has_changed = true;
  } else {
    // Same size, compare element by element
    for(long unsigned int i = 0; i < local_copy_of_leelas_PV.size(); i++) {
      if(local_copy_of_leelas_PV[i].as_string() != local_copy_of_leelas_new_PV[i].as_string()){
	leelas_pv_has_changed = true;
	break;
      }
    }
  }

  if(leelas_pv_has_changed){
    search_stats_->best_move_candidates_mutex.lock();
    search_stats_->Leelas_PV = local_copy_of_leelas_new_PV;
    search_stats_->best_move_candidates_mutex.unlock();    
  }

  if(need_to_restart_thread_one || need_to_restart_thread_two){
    // Change lock and restart the helper threads.
    search_stats_->auxengine_stopped_mutex_.lock();
    if(need_to_restart_thread_one){
      if(!search_stats_->auxengine_stopped_[1]){
	*search_stats_->vector_of_opstreams[1] << "stop" << std::endl; // stop the A/B helper
	search_stats_->auxengine_stopped_[1] = true;
      }
    }
    if(need_to_restart_thread_two){
      if(!search_stats_->auxengine_stopped_[2]){
	*search_stats_->vector_of_opstreams[2] << "stop" << std::endl; // stop the A/B helper
	search_stats_->auxengine_stopped_[2] = true;
      }
    }
    search_stats_->auxengine_stopped_mutex_.unlock();
  }

  if(!only_check_if_Leelas_PV_was_changed){
    if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
    if (current_best_edge_ && !edges.empty()) {
      last_outputted_info_edge_ = current_best_edge_.edge();
    }
    // Cutechess treats each UCI-info line atomically, and if we send multiple lines we have to send the best line last for it to stay on top. This only matters if multipv is set (above one).
    if(max_pv > 1){
      std::reverse(uci_infos.begin(), uci_infos.end());
    }
    uci_responder_->OutputThinkingInfo(&uci_infos);
  }
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  // SharedMutex::Lock lock(nodes_mutex_);
  nodes_mutex_.lock_shared();
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ &&
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo(false);
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      std::vector<ThinkingInfo> info(1);
      info.back().comment =
          "WARNING: Search has reached limit and does not make any progress.";
      uci_responder_->OutputThinkingInfo(&info);
    }
  }
  nodes_mutex_.unlock_shared();  
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_) {
  if (!nps_start_time_) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - *nps_start_time_)
      .count();
}

// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth ? params_.GetOpponentDrawScore()
                       : params_.GetSidetomoveDrawScore()) +
         (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetWhiteDrawDelta()
              : params_.GetBlackDrawDelta());
}

namespace {
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) -
                   value * std::sqrt(node->GetVisitedPolicy());
}

// Faster version for if visited_policy is readily available already.
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score, float visited_pol) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) - value * std::sqrt(visited_pol);
}

inline float ComputeCpuct(const SearchParams& params, uint32_t N,
                          bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}
}  // namespace

std::vector<std::string> Search::GetVerboseStats(Node* node) const {
  assert(node == root_node_ || node->GetParent() == root_node_);
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(params_, node->GetN(), is_root);
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  std::vector<EdgeAndNode> edges;
  for (const auto& edge : node->Edges()) edges.push_back(edge);

  std::sort(edges.begin(), edges.end(),
            [&fpu, &U_coeff, &draw_score](EdgeAndNode a, EdgeAndNode b) {
              return std::forward_as_tuple(
                         a.GetN(), a.GetQ(fpu, draw_score) + a.GetU(U_coeff)) <
                     std::forward_as_tuple(
                         b.GetN(), b.GetQ(fpu, draw_score) + b.GetU(U_coeff));
            });

  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) {
    *oss << pre << std::setw(w) << std::setprecision(p) << v << post;
  };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed;
    print(oss, "", label, " ", 5);
    print(oss, "(", i, ") ", 4);
    *oss << std::right;
    print(oss, "N: ", n, " ", 7);
    print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, p >= 0.99995f ? 1 : 2);
  };
  auto print_stats = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    if (n) {
      print(oss, "(WL: ", sign * n->GetWL(), ") ", 8, 5);
      print(oss, "(D: ", n->GetD(), ") ", 5, 3);
      print(oss, "(M: ", n->GetM(), ") ", 4, 1);
    } else {
      *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
    }
    print(oss, "(Q: ", n ? sign * n->GetQ(sign * draw_score) : fpu, ") ", 8, 5);
  };
  auto print_tail = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    std::optional<float> v;
    if (n && n->IsTerminal()) {
      v = n->GetQ(sign * draw_score);
    } else {
      NNCacheLock nneval = GetCachedNNEval(n);
      if (nneval) v = -nneval->q;
    }
    if (v) {
      print(oss, "(V: ", sign * *v, ") ", 7, 4);
    } else {
      *oss << "(V:  -.----) ";
    }

    if (n) {
      auto [lo, up] = n->GetBounds();
      if (sign == -1) {
        lo = -lo;
        up = -up;
        std::swap(lo, up);
      }
      *oss << (lo == up                                                ? "(T) "
               : lo == GameResult::DRAW && up == GameResult::WHITE_WON ? "(W) "
               : lo == GameResult::BLACK_WON && up == GameResult::DRAW ? "(L) "
                                                                       : "");
    }
  };

  std::vector<std::string> infos;
  const auto m_evaluator = network_->GetCapabilities().has_mlh()
                               ? MEvaluator(params_, node)
                               : MEvaluator();
  for (const auto& edge : edges) {
    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetM(edge, Q);
    std::ostringstream oss;
    oss << std::left;
    // TODO: should this be displaying transformed index?
    print_head(&oss, edge.GetMove(is_black_to_move).as_string(),
               edge.GetMove().as_nn_index(0), edge.GetN(), edge.GetNInFlight(),
               edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", edge.GetU(U_coeff), ") ", 6, 5);
    print(&oss, "(S: ", Q + edge.GetU(U_coeff) + M, ") ", 8, 5);
    print_tail(&oss, edge.node());
    infos.emplace_back(oss.str());
  }

  // Include stats about the node in similar format to its children above.
  std::ostringstream oss;
  print_head(&oss, "node ", node->GetNumEdges(), node->GetN(),
             node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss, node);
  print_tail(&oss, node);
  infos.emplace_back(oss.str());
  return infos;
}

void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  auto move_stats = GetVerboseStats(root_node_);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    uci_responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  for (auto& edge : root_node_->Edges()) {
    if (!(edge.GetMove(played_history_.IsBlackToMove()) == final_bestmove_)) {
      continue;
    }
    if (edge.HasNode()) {
      LOGFILE << "--- Opponent moves after: " << final_bestmove_.as_string();
      for (const auto& line : GetVerboseStats(edge.node())) {
        LOGFILE << line;
      }
    }
  }
}

NNCacheLock Search::GetCachedNNEval(const Node* node) const {
  if (!node) return {};

  std::vector<Move> moves;
  for (; node != root_node_; node = node->GetParent()) {
    moves.push_back(node->GetOwnEdge()->GetMove());
  }
  PositionHistory history(played_history_);
  for (auto iter = moves.rbegin(), end = moves.rend(); iter != end; ++iter) {
    history.Append(*iter);
  }
  const auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
  NNCacheLock nneval(cache_, hash);
  return nneval;
}

void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints, bool update_helper_stuff) {
  hints->Reset();

  Mutex::Lock lock(counters_mutex_);
  // Return early if some other thread already has responded bestmove,
  // or if the root node is not yet expanded.
  if (bestmove_is_sent_ || total_playouts_ + initial_visits_ == 0) {
    return;
  }

  hints->UpdateIndexOfBestEdge(-1);
  if (!stop_.load(std::memory_order_acquire)) {
    if(stopper_->ShouldStop(stats, hints)){
      number_of_skipped_playouts = hints->GetEstimatedRemainingPlayouts();
      FireStopInternal();
    } else {
      // If ShouldStop was rejected due to the most visted move not having the best expected Q, then improve search by boosting exploration of edge of root with the highest expected Q.
      if(hints->GetIndexOfBestEdge() > -1){
	this_edge_has_higher_expected_q_than_the_most_visited_child = hints->GetIndexOfBestEdge();
      }
    }
  } else {
    LOGFILE << "MaybeTriggerStop() not calling stopper_->ShouldStop(), because search is stopped already.";
  }

  if(update_helper_stuff){
    
  // Record the two minimax PVs to their global variables (one to the first divergence, and one to an interesting node in the full length PV.
  // For the first divergence use
  // search_stats_->vector_of_moves_from_root_to_first_minimax_divergence
  // search_stats_->Leelas_minimax_PV_first_divergence_node
  // For the interesting node use
  // search_stats_->Leelas_minimax_PV_some_interesting_node
  // search_stats_->vector_of_moves_from_root_to_some_interesting_minimax_node
  
  // find where Leelas PV diverge from a minimax based PV. store this path so that we can force visits to first node where the minimax based PV diverge.
  std::vector<Move> minimax_pv;
  std::vector<Move> minimax_pv_explore_moves;  
  // Print the MiniMaxQ and the MiniMaxPV to the log
  std::string common_s;
  bool flip = played_history_.IsBlackToMove();
  long unsigned int depth = 1; // needed for drawscore, not sure if it should be 1 or 0 here (for child of root) though.
  Node* n = root_node_;
  Node* some_interesting_node = root_node_;
  bool minimax_divergence_found = false;
  bool some_interesting_node_found = false;
  long unsigned int visits;
  float minimax_q;

  // Create a vector of moves to use for forcing visits to the helper's PV beyond the divergent node.
  // First, just find out the visit distribution over this line in Leelas tree.
  // Secondly, find a suitable point where it would be good to force visits. Currently this is the last node in the PV which has at least 10 more visits than its successor in the PV.

  search_stats_->best_move_candidates_mutex.lock_shared();
  std::vector<Move> helper_PV_from_instance_two = search_stats_->helper_PV_from_instance_two;
  std::vector<Move> helper_PV_from_instance_one = search_stats_->helper_PV_from_instance_one;  
  search_stats_->best_move_candidates_mutex.unlock_shared();
  std::vector<Move> helper_PV_from_instance_two_explore_moves;
  std::string debug_helper_instance_two;
  Node * instance_two_end_node = root_node_;
  std::vector<Move> helper_PV_from_instance_one_explore_moves;
  std::string debug_helper_instance_one;
  Node * instance_one_end_node = root_node_;
  
  nodes_mutex_.lock_shared();
  // MiniMax stuff START
  if(root_node_->GetBestMiniMaxChild() != nullptr){
    minimax_q = -root_node_->GetBestMiniMaxQ();    
  }

  while(some_interesting_node != nullptr &&
	some_interesting_node->GetBestMiniMaxChild() != nullptr &&
	!some_interesting_node_found
	){
    // divergence part
    if(!minimax_divergence_found && n->GetBestMiniMaxChild()->GetOwnEdge()->GetMove(flip).as_string() == GetBestChildNoTemperature(n, depth).edge()->GetMove(flip).as_string()){
      common_s = common_s + n->GetBestMiniMaxChild()->GetOwnEdge()->GetMove(flip).as_string() + " ";
      minimax_pv.push_back(n->GetBestMiniMaxChild()->GetOwnEdge()->GetMove());
      visits = n->GetN();
      n = n->GetBestMiniMaxChild();
    } else {
      if(!minimax_divergence_found){
	minimax_divergence_found = true;
	minimax_pv.push_back(n->GetBestMiniMaxChild()->GetOwnEdge()->GetMove());	
	visits = n->GetN();
	n = n->GetBestMiniMaxChild();
	depth++;
      }
    }
    // some interesting node part
    some_interesting_node = some_interesting_node->GetBestMiniMaxChild();
    // record this move
    minimax_pv_explore_moves.push_back(some_interesting_node->GetOwnEdge()->GetMove());
    // Is this the interesting node? The interesting node is the last node in the PV which has a parent with at least some threshold number of visits more than its BestMiniMaxChild.
    // test if parent has less than child + threshold number of visits. If that is tha case. stop here
    // if(some_interesting_node->GetBestMiniMaxChild()->GetN() + params_.GetAuxEngineExplorationThreshold() > some_interesting_node->GetN()){    
    if(some_interesting_node->GetN() + params_.GetAuxEngineExplorationThreshold() > some_interesting_node->GetParent()->GetN()){    
      // The difference in visits should be larger between the parent and the child, use the parent as entry point.
      some_interesting_node_found = true; // break the loop
      if(minimax_pv_explore_moves.size() > 0){
	minimax_pv_explore_moves.pop_back();
	// backup one generation, it is actually the parent that is the suitable entry point
	some_interesting_node = some_interesting_node->GetParent();
	if(some_interesting_node == root_node_){ // Root is not some interesting node to explore.
	  minimax_pv_explore_moves = {};
	} else {
	  LOGFILE << "Found the second node to explore in the minimax PV at depth " << minimax_pv_explore_moves.size() - 1
		  << " The node currently has " << some_interesting_node->GetN() << " visits, while its parent has "
		  << some_interesting_node->GetParent()->GetN() << " visits.";
	}
      }
    }
    // go deeper
    flip = !flip;      
  }
  // MiniMax stuff STOP
  
  // instance two stuff START
  long unsigned int node_two_found_at_depth = 0;
  flip = played_history_.IsBlackToMove();
  bool interesting_node_two_found = false;
  Node * the_interesting_node_two = root_node_;
  bool unextended_node_two_found = false;
  for(long unsigned int i = 0; i < helper_PV_from_instance_two.size(); i++){
    if(unextended_node_two_found){
      break;
    }
    for (auto& edge : instance_two_end_node->Edges()) {
      if(edge.GetMove() == helper_PV_from_instance_two[i]){
	if(edge.HasNode() && edge.node()->GetN() > 0){	  
	  instance_two_end_node = edge.node();
	  // Where in the PV should we inject visits? Try at the last point where the parent has clearly more visits than the child
	  if(instance_two_end_node->GetN() + params_.GetAuxEngineExplorationThreshold() < instance_two_end_node->GetParent()->GetN() &&
	     !interesting_node_two_found){
	    // Not yet at an interesting node
	    helper_PV_from_instance_two_explore_moves.push_back(edge.GetMove());
	    the_interesting_node_two = instance_two_end_node;
	    node_two_found_at_depth++;
	    debug_helper_instance_two = debug_helper_instance_two + " at depth " + std::to_string(i) + " move: " + edge.GetMove(flip).as_string() + " has " +
	      std::to_string(instance_two_end_node->GetN()) + " visits." ;
	  } else {
	    if(!interesting_node_two_found){
	      // By the next two lines we actually implement an off-by-one error which might be good?
	      interesting_node_two_found = true;
	      // the_interesting_node_two = instance_two_end_node;
	      // helper_PV_from_instance_two_explore_moves.push_back(edge.GetMove());
	      node_two_found_at_depth--;
	      debug_helper_instance_two = debug_helper_instance_two + " Found the interesting node_two at depth " + std::to_string(node_two_found_at_depth) + " move: " + edge.GetMove(flip).as_string() + " has " +
	    std::to_string(instance_two_end_node->GetN()) + " visits." ;
	    } else {
	      // node_two already find in earlier iteration
	      debug_helper_instance_two = debug_helper_instance_two + " at depth " + std::to_string(i) + " move: " + edge.GetMove(flip).as_string() + " has " + std::to_string(instance_two_end_node->GetN()) + " visits." ;
	    }
	  }
	} else {
	  unextended_node_two_found = true;
	  debug_helper_instance_two = debug_helper_instance_two + " Unextended node_two found at depth: " + std::to_string(i);
	}
      }
    }
    flip = !flip;
  }
  // instance two stuff STOP

  // instance one stuff START
  // find the deepest node in Leelas tree on the path defined the by the helpers PV, starting at the first divergence (ie thread 2, not thread 0).

  long unsigned int node_one_found_at_depth = 0;
  flip = played_history_.IsBlackToMove();
  bool interesting_node_one_found = false;
  Node * the_interesting_node_one = root_node_;
  bool unextended_node_one_found = false;
  for(long unsigned int i = 0; i < helper_PV_from_instance_one.size(); i++){
    if(unextended_node_one_found){
      break;
    }
    for (auto& edge : instance_one_end_node->Edges()) {
      if(edge.GetMove() == helper_PV_from_instance_one[i]){
	if(edge.HasNode() && edge.node()->GetN() > 0){	  
	  instance_one_end_node = edge.node();
	  // Where in the PV should we inject visits? Try at the last point where the parent has clearly more visits than the child.
	  if(instance_one_end_node->GetN() + params_.GetAuxEngineExplorationThreshold() < instance_one_end_node->GetParent()->GetN() &&
	     !interesting_node_one_found){
	    // Not yet at an interesting node
	    the_interesting_node_one = instance_one_end_node;	    
	    helper_PV_from_instance_one_explore_moves.push_back(edge.GetMove());
	    node_one_found_at_depth++;
	    debug_helper_instance_one = debug_helper_instance_one + " at depth " + std::to_string(i) + " move: " + edge.GetMove(flip).as_string() + " has " +
	      std::to_string(instance_one_end_node->GetN()) + " visits." ;
	  } else {
	    if(!interesting_node_one_found){
	      // By the next two lines we actually implement an off-by-one error which might be good?
	      interesting_node_one_found = true;
	      // the_interesting_node_one = instance_one_end_node;
	      // helper_PV_from_instance_one_explore_moves.push_back(edge.GetMove());
	      node_one_found_at_depth--;
	      debug_helper_instance_one = debug_helper_instance_one + " Found the interesting node_one at depth " + std::to_string(node_one_found_at_depth) + " move: " + edge.GetMove(flip).as_string() + " has " +
		std::to_string(the_interesting_node_one->GetN()) + " visits." ;
	    } else {
	      // node_one already found in earlier iteration
	      debug_helper_instance_one = debug_helper_instance_one + " at depth " + std::to_string(i) + " move: " + edge.GetMove(flip).as_string() + " has " + std::to_string(instance_one_end_node->GetN()) + " visits." ;
	    }
	  }
	} else {
	  unextended_node_one_found = true;
	  debug_helper_instance_one = debug_helper_instance_one + " Unextended node_one found at depth: " + std::to_string(i);
	}
      }
    }
    flip = !flip;
  }
  // instance one stuff STOP
  
  nodes_mutex_.unlock_shared();  

  search_stats_->best_move_candidates_mutex.lock(); // for reading search_stats_->winning_ and the other
  // Record the interesting node in the PV of thread two.
  if(helper_PV_from_instance_two_explore_moves.size() > 0 &&
     search_stats_->helper_PV_from_instance_two_explore_moves != helper_PV_from_instance_two_explore_moves
     ){
    search_stats_->helper_PV_from_instance_two_explore_moves = helper_PV_from_instance_two_explore_moves;
    search_stats_->helper_PV_from_instance_two_explore_node = the_interesting_node_two;
    LOGFILE << "Summary status of the PV from thread two after finishing a batch. The PV is of length " << helper_PV_from_instance_two.size()
	    << " and the selected node_two found at depth " << node_two_found_at_depth << " with " << the_interesting_node_two->GetN()
	    << " visits. " << debug_helper_instance_two << " Debugging info on the selected node_two: " << the_interesting_node_two->DebugString()
	    << " and the edge leading to it: " << the_interesting_node_two->GetOwnEdge()->DebugString();
  }
  // Record the interesting node in the PV of thread one.
  if(helper_PV_from_instance_one_explore_moves.size() > 0 &&
     search_stats_->helper_PV_from_instance_one_explore_moves != helper_PV_from_instance_one_explore_moves
     ){
    search_stats_->helper_PV_from_instance_one_explore_moves = helper_PV_from_instance_one_explore_moves;
    search_stats_->helper_PV_from_instance_one_explore_node = the_interesting_node_one;
    LOGFILE << "Summary status of the PV from thread one after finishing a batch. The PV is of length " << helper_PV_from_instance_one.size()
	    << " and the selected node_one found at depth " << node_one_found_at_depth << " with " << the_interesting_node_one->GetN()
	    << " visits. " << debug_helper_instance_one << " Debugging info on the selected node_one: " << the_interesting_node_one->DebugString()
	    << " and the edge leading to it: " << the_interesting_node_one->GetOwnEdge()->DebugString();
  }
  // If a minimax divergence were found, record it.
  if(minimax_divergence_found){
    // only write if it has changed
    if(search_stats_->Leelas_minimax_PV_first_divergence_node != n){
      LOGFILE << "Q based on MiniMax: " << minimax_q << ". Divergence found at depth " << minimax_pv.size()-1
	      << " common PV is " << common_s << " first move to explore more since is on the minimax path is (from whites perspective) " << minimax_pv[minimax_pv.size()-1].as_string()
	      << " the node here has " << visits << " visits.";
      search_stats_->vector_of_moves_from_root_to_first_minimax_divergence = minimax_pv;
      search_stats_->Leelas_minimax_PV_first_divergence_node = n;
      // Also query the helper about this node, if not already done.
      if(n->GetAuxEngineMove() == 0xffff &&
	 !n->IsTerminal() &&
	 n->HasChildren() &&
	 // These last two conditions are rather expensive to evaluate, which is why they must come last
	 params_.GetAuxEngineFile() != ""
	 ){
	LOGFILE << "Querying the helper about the first divergence in the minimax PV";
	// note nested lock best_move_candidates and auxengine_mutex
	search_stats_->auxengine_mutex_.lock();
	search_stats_->persistent_queue_of_nodes.push(n);
	auxengine_cv_.notify_one(); // unnecessary?
	n->SetAuxEngineMove(0xfffe); // magic for pending. Lock needed here? Nah.	
	search_stats_->auxengine_mutex_.unlock();
      }
    }
  } else {
    if(search_stats_->vector_of_moves_from_root_to_first_minimax_divergence.size() > 0){
      // There was a divergence, but it is gone now, remove the obsolete PV.
      LOGFILE << "There was a divergence, but it is gone now, removing the obsolete PV.";
      search_stats_->vector_of_moves_from_root_to_first_minimax_divergence = {};
    }
  }

  // Repeat for the second minimax node
  // If a minimax divergence were found, record it.
  if(minimax_pv_explore_moves.size() > 0){
    // only write if it has changed
    if(search_stats_->Leelas_minimax_PV_some_interesting_node != some_interesting_node){
      search_stats_->vector_of_moves_from_root_to_some_interesting_minimax_node = minimax_pv_explore_moves;
      search_stats_->Leelas_minimax_PV_some_interesting_node = some_interesting_node;
      // Also query the helper about this node, if not already done.
      if(some_interesting_node->GetAuxEngineMove() == 0xffff &&
	 !some_interesting_node->IsTerminal() &&
	 some_interesting_node->HasChildren() &&
	 // These last two conditions are rather expensive to evaluate, which is why they must come last
	 params_.GetAuxEngineFile() != ""
	 ){
	LOGFILE << "Querying the helper about some_interesting_node in the minimax PV";
	// note nested lock best_move_candidates and auxengine_mutex
	search_stats_->auxengine_mutex_.lock();
	search_stats_->persistent_queue_of_nodes.push(some_interesting_node);
	auxengine_cv_.notify_one(); // unnecessary?
	some_interesting_node->SetAuxEngineMove(0xfffe); // magic for pending. Lock needed here? Nah.
	search_stats_->auxengine_mutex_.unlock();
      }
    }
  } else {
    if(search_stats_->vector_of_moves_from_root_to_some_interesting_minimax_node.size() > 0){
      // There was an interesting node, but it is not interesting anymore and no new node is either.
      LOGFILE << "There was some interesting minimax node, but it is gone now, removing the obsolete PV.";
      search_stats_->vector_of_moves_from_root_to_some_interesting_minimax_node = {};
    }
  }
  
  search_stats_->best_move_candidates_mutex.unlock();

  } // end of if update_helper_stuff

  bool this_tread_triggered_stop = false;
  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {

    this_tread_triggered_stop = true;
    search_stats_->auxengine_stopped_mutex_.lock();
    // Check the status for each thread, and act accordingly
    // give the helper engines some slack, perhaps they were started just a millisecond ago.
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    for(long unsigned int i = 0; i < search_stats_->auxengine_stopped_.size() ; i++){
      if(!search_stats_->auxengine_stopped_[i]){
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "MaybeTriggerStop() Stopping the A/B helper Start for thread=" << i << " Start.";
	*search_stats_->vector_of_opstreams[i] << "stop" << std::endl; // stop the A/B helper
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "MaybeTriggerStop() Stopping the A/B helper for thread=" << i << " Stop.";
	search_stats_->auxengine_stopped_[i] = true;
      } else {
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "MaybeTriggerStop() Not stopping the A/B helper for thread=" << i << ".";      	
      }
    }
    search_stats_->auxengine_stopped_mutex_.unlock();

    nodes_mutex_.lock_shared();
    // veto if the move Leela prefers is a blunder
    search_stats_->best_move_candidates_mutex.lock(); // for reading search_stats_->winning_ and the other
    
    if(! search_stats_->winning_){

       // Let the helper play the endgame.
      ChessBoard board = played_history_.Last().GetBoard();
      if(board.ours().count() + board.theirs().count() < 15){
	LOGFILE << "Let the helper play the endgame. Less than 15 pieces left now.";
	search_stats_->winning_ = true;
      }

      if(search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child > 0){
	if(search_stats_->Leelas_PV[0].as_string() != search_stats_->winning_move_.as_string()){	   
	  if(params_.GetAuxEngineVerbosity() >= 2){
	    LOGFILE << "leelas preferred child differs from the move recommended by the helper. \n"
		    << "Helper evaluates Leelas preferred move to: " << search_stats_->helper_eval_of_leelas_preferred_child << " based on " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child << " nodes.\n"
	            << "Helper evaluates its own prefered move to: " << search_stats_->helper_eval_of_helpers_preferred_child << " based on " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_helpers_preferred_child << " nodes.\n"
		    << "Helper evaluates root to: " << search_stats_->helper_eval_of_root << " based on " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_root << " nodes.";
	  }
	  if(search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child > 1000000 &&
	     search_stats_->number_of_nodes_in_support_for_helper_eval_of_helpers_preferred_child > 1000000){
	    int centipawn_diff = search_stats_->helper_eval_of_helpers_preferred_child - search_stats_->helper_eval_of_leelas_preferred_child;
	    if(
	       // improve play
	       // improve play in worse positions
	       (search_stats_->helper_eval_of_leelas_preferred_child < -30 && centipawn_diff > 10) ||
	       // improve play in better positions
	       (search_stats_->helper_eval_of_leelas_preferred_child > 30 && centipawn_diff > 10) ||
	       // save the win
	       (search_stats_->helper_eval_of_root > 90 && centipawn_diff > 9) ||
	       
	       // save the draw
	       // 90 110
	       (search_stats_->helper_eval_of_leelas_preferred_child < -90 && centipawn_diff > 9) ||
	       // save the win
	       (search_stats_->helper_eval_of_root > 110 && centipawn_diff > 8) ||
	       // 120 135
	       // save the draw	       
	       (search_stats_->helper_eval_of_leelas_preferred_child < -110 && centipawn_diff > 8) ||
	       // save the win	       
	       (search_stats_->helper_eval_of_root > 120 && centipawn_diff > 7) ||
	       // 140 152
	       (search_stats_->helper_eval_of_leelas_preferred_child < -120 && centipawn_diff > 7)
	       ){
	      // print the move in rotated mode
	      bool flip = played_history_.IsBlackToMove();
	      Move m_helper;
	      Move m_leela;	      
	      Move::ParseMove(&m_helper, search_stats_->winning_move_.as_string(), flip);
	      Move::ParseMove(&m_leela, search_stats_->Leelas_PV[0].as_string(), flip);
	      if (params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "Trying to save a draw/win, helper eval of root: " << search_stats_->helper_eval_of_root << " helper recommended move " << m_helper.as_string() << ". Number of nodes in support for the root node eval: " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_root << " The helper eval of leelas preferred move: " << search_stats_->helper_eval_of_leelas_preferred_child << " Leela prefers the move: " << m_leela.as_string() << ", nodes in support for the eval of leelas preferred move: " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child << " helper eval of helpers preferred move: " << search_stats_->helper_eval_of_helpers_preferred_child << " Number of nodes in support for helpers preferred child: " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_helpers_preferred_child << " centipawn diff between helpers and leelas line, according to the helper: " << std::abs(search_stats_->helper_eval_of_helpers_preferred_child - search_stats_->helper_eval_of_leelas_preferred_child) << ".";
	      search_stats_->stop_a_blunder_ = true;
	      if(search_stats_->helper_eval_of_root > 140){
		search_stats_->save_a_win_ = true;
	      } else {
		search_stats_->save_a_win_ = false;
	      }
	    } else {
	      // Too small differences, but which one had the better move (according to the helper)
	      if(centipawn_diff > 0){
		LOGFILE << "Helper had the better move by " << centipawn_diff << " cp (according to itself).";
	      } 
	      if(centipawn_diff < 0){
		LOGFILE << "Leela had the better move by " << std::abs(centipawn_diff) << " cp (according to the helper).";
	      } 
	    }
	  } else {
	    LOGFILE << "Too few nodes in support of the helpers eval of Leelas preferred child: " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child << " not considering to veto Leelas choice.";
	  }
	} else {
	  // They agree.
	  if(params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "Leela agree with the helper about the best move: " << search_stats_->Leelas_PV[0].as_string() << ". The root explorer evaluates root to: " << search_stats_->helper_eval_of_root << " based on " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_root << " nodes.";
	}
      } else {
	if(search_stats_->number_of_nodes_in_support_for_helper_eval_of_helpers_preferred_child > 0){
	  LOGFILE << "number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child is not enough for vetoing. nodes: "
		  << search_stats_->number_of_nodes_in_support_for_helper_eval_of_helpers_preferred_child << " eval for helpers PV "
		  << search_stats_->helper_eval_of_helpers_preferred_child << " Helper evaluates Leelas preferred move to: "
		  << search_stats_->helper_eval_of_leelas_preferred_child << " based on " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child << " nodes.\n"
		  << "Helper evaluates its own prefered move to: " << search_stats_->helper_eval_of_helpers_preferred_child << " based on " << search_stats_->number_of_nodes_in_support_for_helper_eval_of_helpers_preferred_child;
	} 
      }
    } else {
      LOGFILE << "Autopilot is on.";
    }
	  
    search_stats_->best_move_candidates_mutex.unlock();
    nodes_mutex_.unlock_shared();    

    if(params_.GetAuxEngineVerbosity() >= 3){
      LOGFILE << "Finished vetoing stuff";
    }

    SendUciInfo(false);
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
    this_edge_has_higher_expected_q_than_the_most_visited_child = -1;
    // if we set stop_a_blunder, then that has had its effect by now, reset it so it won't affect next move
    search_stats_->best_move_candidates_mutex.lock();
    if(search_stats_->stop_a_blunder_){
      if(params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Resetting search_stats_->stop_a_blunder_ to false";
      search_stats_->stop_a_blunder_ = false;
    }
    search_stats_->best_move_candidates_mutex.unlock();    
  } else {
    SendUciInfo(true); // Just update Leelas PV, don't send any UCI info.
  }

  // Use a 0 visit cancel score update to clear out any cached best edge, as
  // at the next iteration remaining playouts may be different.
  // TODO(crem) Is it really needed?
  root_node_->CancelScoreUpdate(0);

  // Confirm that this function exited successfully when stop was triggered.
  if (this_tread_triggered_stop) {
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Finished MaybeTriggerStop(): stopped search and successfully shutdown.";
  } else {
    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Finished MaybeTriggerStop() finished, not stopping search yet.";
  }
}

// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
Eval Search::GetBestEval(Move* move, bool* is_terminal) const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();
  if (!root_node_->HasChildren()) return {parent_wl, parent_d, parent_m};
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_, 0);
  if (move) *move = best_edge.GetMove(played_history_.IsBlackToMove());
  if (is_terminal) *is_terminal = best_edge.IsTerminal();
  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.GetM(parent_m - 1) + 1};
}

std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}

std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}

void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown();
  bestmove_is_sent_ = old_sent;
}

// Computes the best move, maybe with temperature (according to the settings).
void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_)
    REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (root_node_->GetN() == 0) return;
  if (!root_node_->HasChildren()) return;

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves = played_history_.Last().GetGamePly() / 2;

  if (cutoff_move && (moves + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature && decay_moves) {
    if (moves >= decay_delay_moves + decay_moves) {
      temperature = 0.0;
    } else if (moves >= decay_delay_moves) {
      temperature *=
          static_cast<float>(decay_delay_moves + decay_moves - moves) /
          decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }

  auto bestmove_edge = temperature
                           ? GetBestRootChildWithTemperature(temperature)
                           : GetBestChildNoTemperature(root_node_, 0);
  final_bestmove_ = bestmove_edge.GetMove(played_history_.IsBlackToMove());

  if (bestmove_edge.GetN() > 0 && bestmove_edge.node()->HasChildren()) {
    final_pondermove_ = GetBestChildNoTemperature(bestmove_edge.node(), 1)
                            .GetMove(!played_history_.IsBlackToMove());
  }
}

// Returns @count children with most visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const {
  // Even if Edges is populated at this point, its a race condition to access
  // the node, so exit quickly.
  if (parent->GetN() == 0) return {};
  const bool is_odd_depth = (depth % 2) == 1;
  bool vetoing_already_announced = false; // only print the message about vetoing once, not once per edge.
  const float draw_score = GetDrawScore(is_odd_depth);
  const bool select_move_by_q = params_.GetQBasedMoveSelection() && (stop_.load(std::memory_order_acquire) || parent->GetN() > 10000); // GetBestChildrenNoTemperature is called by GetBestChildNoTemperature(), which in turn is called by PreExtend..() To enhance performance only do the beta calculations when needed.
  // inactivating Q based best child when parent has 10000 visits. This can potentially be a large change in playing style.
  // const bool select_move_by_q = params_.GetQBasedMoveSelection() && (stop_.load(std::memory_order_acquire)); // GetBestChildrenNoTemperature is called by GetBestChildNoTemperature(), which in turn is called by PreExtend..() To enhance performance only do the beta calculations when needed.
  const float beta_prior = pow(parent->GetN() + number_of_skipped_playouts, params_.GetMoveSelectionVisitsScalingPower());
  // float beta_prior; // only calculate a value for this when needed.
  // if(select_move_by_q){
  //   beta_prior = pow(parent->GetN() + number_of_skipped_playouts, params_.GetMoveSelectionVisitsScalingPower());
  // }
  number_of_skipped_playouts = 0; // if search runs out of time, this is the correct number, and if search is stopped early this value will be overwritten.
  // Best child is selected using the following criteria:
  // * Prefer shorter terminal wins / avoid shorter terminal losses.
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  std::vector<EdgeAndNode> edges;
  for (auto& edge : parent->Edges()) {
    if (parent == root_node_ && !root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    edges.push_back(edge);
  }
  const auto middle = (static_cast<int>(edges.size()) > count)
                          ? edges.begin() + count
                          : edges.end();

  bool winning_ = false;
  Move winning_move_;

  // This function is called very often, only check for winning when
  // it is move selection time. TODO, add a boolean parameter to this
  // function, and use that from SendUCI info, that way we always
  // display correct move ordering.
  if(stop_.load(std::memory_order_acquire)){
    search_stats_->best_move_candidates_mutex.lock_shared();
    winning_ = search_stats_->winning_ || search_stats_->stop_a_blunder_;
    if (winning_){
      winning_move_ = search_stats_->winning_move_;
      std::string state;
      if(search_stats_->winning_){
	state="is winning";
      } else {
	state="stops a blunder";
      }
      if (params_.GetAuxEngineVerbosity() >= 2 && parent == root_node_ && !vetoing_already_announced) LOGFILE << "The move: " << winning_move_.as_string() << " will override Q and N based comparisons, since the helper claims it " << state << "." ;
      vetoing_already_announced = true;
    }
    search_stats_->best_move_candidates_mutex.unlock_shared();
  }
  
  std::partial_sort(
      edges.begin(), middle, edges.end(),
      [draw_score, beta_prior, select_move_by_q, winning_, winning_move_](const auto& a, const auto& b) {
        // The function returns "true" when a is preferred to b.

        // Lists edge types from less desirable to more desirable.
        enum EdgeRank {
          kTerminalLoss,
          kTablebaseLoss,
          kNonTerminal,  // Non terminal or terminal draw.
          kTablebaseWin,
          kTerminalWin,
        };

        auto GetEdgeRank = [](const EdgeAndNode& edge) {
          // This default isn't used as wl only checked for case edge is
          // terminal.
          const auto wl = edge.GetWL(0.0f);
          // Not safe to access IsTerminal if GetN is 0.
          if (edge.GetN() == 0 || !edge.IsTerminal() || !wl) {
            return kNonTerminal;
          }
          if (edge.IsTbTerminal()) {
            return wl < 0.0 ? kTablebaseLoss : kTablebaseWin;
          }
          return wl < 0.0 ? kTerminalLoss : kTerminalWin;
        };

        // If moves have different outcomes, prefer better outcome.
        const auto a_rank = GetEdgeRank(a);
        const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank;

        // If both are terminal draws, try to make it shorter.
        // Not safe to access IsTerminal if GetN is 0.
        if (a_rank == kNonTerminal && a.GetN() != 0 && b.GetN() != 0 &&
            a.IsTerminal() && b.IsTerminal()) {
          if (a.IsTbTerminal() != b.IsTbTerminal()) {
            // Prefer non-tablebase draws.
            return a.IsTbTerminal() < b.IsTbTerminal();
          }
          // Prefer shorter draws.
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Neither is terminal, use standard rule.
        if (a_rank == kNonTerminal) {

	  // if the helper engine claims a win (or saves a draw), trust it.
	  if (winning_){
	    if(winning_move_ == a.GetMove()){
	      return(true);
	    }
	    if(winning_move_ == b.GetMove()){
	      return(false);
	    }
	  }

	  if(select_move_by_q){
	    // the beta_prior is constant and equals:
	    // pow(parent->GetN(), params_.GetMoveSelectionVisitsScalingPower());
	    float alpha_prior = 0.0f;

	    float winrate_a = (a.GetQ(0.0f, draw_score) + 1) * 0.5;
	    int visits_a = a.GetN();
	    float alpha_a = winrate_a * visits_a + alpha_prior;
	    float beta_a = visits_a - alpha_a + beta_prior;
	    float E_a = alpha_a / (alpha_a + beta_a);

	    float winrate_b = (b.GetQ(0.0f, draw_score) + 1) * 0.5;
	    int visits_b = b.GetN();
	    float alpha_b = winrate_b * visits_b + alpha_prior;
	    float beta_b = visits_b - alpha_b + beta_prior;
	    float E_b = alpha_b / (alpha_b + beta_b);

	    if (E_a != E_b) return(E_a > E_b);
	  }

          // Prefer largest playouts then eval then prior.
          if (a.GetN() != b.GetN()) return a.GetN() > b.GetN();
          // Default doesn't matter here so long as they are the same as either
          // both are N==0 (thus we're comparing equal defaults) or N!=0 and
          // default isn't used.
          if (a.GetQ(0.0f, draw_score) != b.GetQ(0.0f, draw_score)) {
            return a.GetQ(0.0f, draw_score) > b.GetQ(0.0f, draw_score);
          }
          return a.GetP() > b.GetP();
        }

        // Both variants are winning, prefer shortest win.
        if (a_rank > kNonTerminal) {
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Both variants are losing, prefer longest losses.
        return a.GetM(0.0f) > b.GetM(0.0f);
      });

  if (count < static_cast<int>(edges.size())) {
    edges.resize(count);
  }
  return edges;
}

// Returns a child with most visits.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}

// Returns a child of a root chosen according to weighted-by-temperature visit
// count.
EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const {
  // Root is at even depth.
  const float draw_score = GetDrawScore(/* is_odd_depth= */ false);

  std::vector<float> cumulative_sums;
  float sum = 0.0;
  float max_n = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -1.0f;
  const float fpu =
      GetFpu(params_, root_node_, /* is_root= */ true, draw_score);

  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetN() + offset > max_n) {
      max_n = edge.GetN() + offset;
      max_eval = edge.GetQ(fpu, draw_score);
    }
  }

  // TODO(crem) Simplify this code when samplers.h is merged.
  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    sum += std::pow(
        std::max(0.0f,
                 (max_n <= 0.0f
                      ? edge.GetP()
                      : ((static_cast<float>(edge.GetN()) + offset) / max_n))),
        1 / temperature);
    cumulative_sums.push_back(sum);
  }
  assert(sum);

  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    if (idx-- == 0) return edge;
  }
  assert(false);
  return {};
}

void Search::StartThreads(size_t how_many) {
  thread_count_.store(how_many, std::memory_order_release);
  Mutex::Lock lock(threads_mutex_);
  OpenAuxEngine();
  // First thread is a watchdog thread.
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  for (size_t i = 0; i < how_many; i++) {
    threads_.emplace_back([this, i]() {
      SearchWorker worker(this, params_, i);
      worker.RunBlocking();
    });
  }
  LOGFILE << "Search started. "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time_)
                 .count()
          << "ms already passed.";
}

void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}

bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

void Search::PopulateCommonIterationStats(IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();

  SharedMutex::SharedLock nodes_lock(nodes_mutex_);
  {
    Mutex::Lock counters_lock(counters_mutex_);
    stats->time_since_first_batch = GetTimeSinceFirstBatch();
    if (!nps_start_time_ && total_playouts_ > 0) {
      nps_start_time_ = std::chrono::steady_clock::now();
    }
  }
  stats->total_nodes = total_playouts_ + initial_visits_;
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  stats->edge_n.clear();
  stats->q.clear();
  stats->win_found = false;
  stats->may_resign = true;
  stats->num_losing_edges = 0;
  stats->move_selection_visits_scaling_power = params_.GetMoveSelectionVisitsScalingPower();
  stats->override_PUCT_node_budget_threshold = params_.GetOverridePUCTNodeBudgetThreshold();

  // bool found_the_edge = false;
  // search_stats_->best_move_candidates_mutex.lock();
  // LOGFILE << "1.";
  // if(search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child > 0){
  //   LOGFILE << "1.5.";    
  //   if(search_stats_->Leelas_preferred_child_node_ != nullptr){
  //     LOGFILE << "2.";
  //     if(search_stats_->Leelas_preferred_child_node_->GetOwnEdge() != nullptr &&
  // 	 search_stats_->number_of_nodes_in_support_for_helper_eval_of_root > 0 &&
  // 	 search_stats_->winning_move_.as_string() != search_stats_->Leelas_preferred_child_node_->GetOwnEdge()->GetMove().as_string()){
  // 	LOGFILE << "3.";      
  // 	stats->agreement_between_Leela_and_helper = false;
  // 	stats->Leelas_preferred_child_node_visits = search_stats_->Leelas_preferred_child_node_->GetN();
  // 	stats->helper_eval_of_root = search_stats_->helper_eval_of_root;
  // 	stats->helper_eval_of_leelas_preferred_child = search_stats_->helper_eval_of_leelas_preferred_child;
  // 	// Find the node corresponding to the helper recommended move
  // 	int index = 0;
  // 	for (auto& edge : root_node_->Edges()) {
  // 	  if(edge.GetMove() == search_stats_->winning_move_){
  // 	    stats->helper_recommended_node_visits = edge.node()->GetN();
  // 	    stats->helper_recommended_index = index;
  // 	    found_the_edge = true;
  // 	    break;
  // 	  }
  // 	  index++;
  // 	}
  //     }
  //   }
  // } else {
  //   stats->agreement_between_Leela_and_helper = true;    
  // }
  // search_stats_->best_move_candidates_mutex.unlock();
  // if(! stats->agreement_between_Leela_and_helper &&
  //    ! found_the_edge){
  //   // Sanity check failed The edge was not found, we can't use it to stop pruning
  //   LOGFILE << "Sanity check failed, the edge was not found, we can't use it to stop pruning";
  //   stats->agreement_between_Leela_and_helper = true;
  // }

  stats->agreement_between_Leela_and_helper = true;

  stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNormal;

  // If root node hasn't finished first visit, none of this code is safe.
  if (root_node_->GetN() > 0) {
    const auto draw_score = GetDrawScore(true);
    const float fpu =
        GetFpu(params_, root_node_, /* is_root_node */ true, draw_score);
    float max_q_plus_m = -1000;
    uint64_t max_n = 0;
    bool max_n_has_max_q_plus_m = true;
    const auto m_evaluator = network_->GetCapabilities().has_mlh()
                                 ? MEvaluator(params_, root_node_)
                                 : MEvaluator();
    for (const auto& edge : root_node_->Edges()) {
      const auto n = edge.GetN();
      const auto q = edge.GetQ(fpu, draw_score);
      const auto m = m_evaluator.GetM(edge, q);
      const auto q_plus_m = q + m;
      stats->q.push_back(q);
      stats->edge_n.push_back(n);
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) > 0.0f) {
        stats->win_found = true;
      }
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) < 0.0f) {
        stats->num_losing_edges += 1;
      }
      // If game is resignable, no need for moving quicker. This allows
      // proving mate when losing anyway for better score output.
      // Hardcoded resign threshold, because there is no available parameter.
      if (n > 0 && q > -0.98f) {
        stats->may_resign = false;
      }
      if (max_n < n) {
        max_n = n;
        max_n_has_max_q_plus_m = false;
      }
      if (max_q_plus_m <= q_plus_m) {
        max_n_has_max_q_plus_m = (max_n == n);
        max_q_plus_m = q_plus_m;
      }
    }
    if (!max_n_has_max_q_plus_m) {
      stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNeedMoreTime;
    }
  }
}

void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  StoppersHints hints;
  IterationStats stats;
  while (true) {
    PopulateCommonIterationStats(&stats);
    MaybeTriggerStop(stats, &hints, false);
    MaybeOutputInfo();

    constexpr auto kMaxWaitTimeMs = 100;
    constexpr auto kMinWaitTimeMs = 1;

    Mutex::Lock lock(counters_mutex_);
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;

    auto remaining_time = hints.GetEstimatedRemainingTimeMs();
    if (remaining_time > kMaxWaitTimeMs) remaining_time = kMaxWaitTimeMs;
    if (remaining_time < kMinWaitTimeMs) remaining_time = kMinWaitTimeMs;
    // There is no real need to have max wait time, and sometimes it's fine
    // to wait without timeout at all (e.g. in `go nodes` mode), but we
    // still limit wait time for exotic cases like when pc goes to sleep
    // mode during thinking.
    // Minimum wait time is there to prevent busy wait and other threads
    // starvation.
    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time),
        [this]() { return stop_.load(std::memory_order_acquire); });
  }
  LOGFILE << "End a watchdog thread.";
}

void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
  auxengine_cv_.notify_all();  
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Stopping search due to `stop` uci command.";
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "from Stop() About to enter AuxWait()";
  AuxWait();  // This can take some time during which we are not ready to respond readyok, so for now increase timemargin.
  // When would AuxWait() actually take long time? if MaybetriggerStop() somehow fails to detect that a AuxEngingeWorker-thread actually should be stopped,
  // or if sending `stop` to the helper engine somehow does not actually stop it, but both of these should never happen, so I don't think AuxWait() takes
  // substantial time any more. The AuxEngineWorker threads should all be idle at this point (since MaybeTriggerStop() now waits for them before exiting).
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "from Stop() AuxWait() returned";  

}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

void Search::CancelSharedCollisions() REQUIRES(nodes_mutex_) {
  for (auto& entry : shared_collisions_) {
    Node* node = entry.first;
    for (node = node->GetParent(); node != root_node_->GetParent();
         node = node->GetParent()) {
      node->CancelScoreUpdate(entry.second);
    }
  }
  shared_collisions_.clear();
}

Search::~Search() {
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "About to destroy search.";
  Abort();
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "In the search destructor about to Wait().";
  Wait();
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "In the search destructor about to get a lock.";
  nodes_mutex_.lock_shared();
  CancelSharedCollisions();
  nodes_mutex_.unlock_shared();
  AuxWait();
  LOGFILE << "Search destroyed.";
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

void SearchWorker::RunTasks(int tid) {
  while (true) {
    PickTask* task = nullptr;
    int id = 0;
    {
      int spins = 0;
      while (true) {
        int nta = tasks_taken_.load(std::memory_order_acquire);
        int tc = task_count_.load(std::memory_order_acquire);
        if (nta < tc) {
          int val = 0;
          if (task_taking_started_.compare_exchange_weak(
                  val, 1, std::memory_order_acq_rel,
                  std::memory_order_relaxed)) {
            nta = tasks_taken_.load(std::memory_order_acquire);
            tc = task_count_.load(std::memory_order_acquire);
            // We got the spin lock, double check we're still in the clear.
            if (nta < tc) {
              id = tasks_taken_.fetch_add(1, std::memory_order_acq_rel);
              task = &picking_tasks_[id];
              task_taking_started_.store(0, std::memory_order_release);
              break;
            }
            task_taking_started_.store(0, std::memory_order_release);
          }
          SpinloopPause();
          spins = 0;
          continue;
        } else if (tc != -1) {
          spins++;
          if (spins >= 512) {
            std::this_thread::yield();
            spins = 0;
          } else {
            SpinloopPause();
          }
          continue;
        }
        spins = 0;
        // Looks like sleep time.
        Mutex::Lock lock(picking_tasks_mutex_);
        // Refresh them now we have the lock.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (tc != -1) continue;
        if (nta >= tc && exiting_) return;
        task_added_.wait(lock.get_raw());
        // And refresh again now we're awake.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (nta >= tc && exiting_) return;
      }
    }
    if (task != nullptr) {
      switch (task->task_type) {
        case PickTask::kGathering: {
          PickNodesToExtendTask(task->start, task->base_depth,
                                task->collision_limit, task->moves_to_base,
                                &(task->results), &(task_workspaces_[tid]), task->probability_of_best_path, task->distance_from_best_path, 0);
          break;
        }
        case PickTask::kProcessing: {
          ProcessPickedTask(task->start_idx, task->end_idx,
                            &(task_workspaces_[tid]));
          break;
        }
      }
      picking_tasks_[id].complete = true;
      completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}

void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration(search_->network_->NewComputation());
  if (params_.GetMaxConcurrentSearchers() != 0) {
    while (true) {
      // If search is stop, we've not gathered or done anything and we don't
      // want to, so we can safely skip all below. But make sure we have done
      // at least one iteration.
      if (search_->stop_.load(std::memory_order_acquire) &&
          search_->GetTotalPlayouts() + search_->initial_visits_ > 0) {
        return;
      }
      int available =
          search_->pending_searchers_.load(std::memory_order_acquire);
      if (available > 0 &&
          search_->pending_searchers_.compare_exchange_weak(
              available, available - 1, std::memory_order_acq_rel)) {
        break;
      }
      // This is a hard spin lock to reduce latency but at the expense of busy
      // wait cpu usage. If search worker count is large, this is probably a bad
      // idea.
    }
  }

  // 1.5 Extend tree with nodes using PV of a/b helper, and add the new
  // nodes to the minibatch
  const std::shared_ptr<Search::adjust_policy_stats> foo = PreExtendTreeAndFastTrackForNNEvaluation();
  // std::queue<std::vector<Node*>> queue_of_vector_of_nodes_from_helper_added_by_this_thread = PreExtendTreeAndFastTrackForNNEvaluation();

  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << std::this_thread::get_id() << " PreExtendTreeAndFastTrackForNNEvaluation() finished in ExecuteOneIteration().";
  // 2. Gather minibatch.

  int number_of_nodes_already_added = minibatch_.size();
  GatherMinibatch2(number_of_nodes_already_added);

  task_count_.store(-1, std::memory_order_release);
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);
  // if (params_.GetAuxEngineVerbosity() >= 3) LOGFILE << "GatherMinibatch2() finished in ExecuteOneIteration().";
  
  // 2b. Collect collisions.
  CollectCollisions();

  // 3. Prefetch into cache.
  MaybePrefetchIntoCache();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }

  // 4. Run NN computation.
  LOGFILE << "Minibatch size: " << minibatch_.size();
  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);
  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << std::this_thread::get_id() << " RunNNComputation() finished in ExecuteOneIteration().";
  
  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();
  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << std::this_thread::get_id() << " FetchMinibatchResults() finished in ExecuteOneIteration().";
  
  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();
  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << std::this_thread::get_id() << " DoBackupUpdate() finished in ExecuteOneIteration().";

  MaybeAdjustPolicyForHelperAddedNodes(foo);
  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << std::this_thread::get_id() << " MaybeAdjustPolicyForHelperAddedNodes() finished in ExecuteOneIteration().";

  // 7. Update the Search's status and progress information.
  UpdateCounters();

  // if (params_.GetAuxEngineVerbosity() >= 2) LOGFILE << std::this_thread::get_id() << " finished one full iteration.";  

  // If required, waste time to limit nps.
  if (params_.GetNpsLimit() > 0) {
    while (search_->IsSearchActive()) {
      int64_t time_since_first_batch_ms = 0;
      {
        Mutex::Lock lock(search_->counters_mutex_);
        time_since_first_batch_ms = search_->GetTimeSinceFirstBatch();
      }
      if (time_since_first_batch_ms <= 0) {
        time_since_first_batch_ms = search_->GetTimeSinceStart();
      }
      auto nps = search_->GetTotalPlayouts() * 1e3f / time_since_first_batch_ms;
      if (nps > params_.GetNpsLimit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        break;
      }
    }
  }
}

// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> computation) {
  computation_ = std::make_unique<CachingComputation>(std::move(computation),
                                                      search_->cache_);
  computation_->Reserve(params_.GetMiniBatchSize());
  minibatch_.clear();
  minibatch_.reserve(2 * params_.GetMiniBatchSize());
}

  void SearchWorker::PreExtendTreeAndFastTrackForNNEvaluation_inner(Node * my_node, std::vector<lczero::Move> my_moves, int ply, int nodes_added, int source, std::vector<Node*>* nodes_from_helper_added_by_this_PV, int amount_of_support, float probability_of_best_path) {

  bool black_to_move = ! search_->played_history_.IsBlackToMove() ^ (ply % 2 == 0);
  bool edge_found = false;

  // // Check if search is stopped.
  // if(search_->stop_.load(std::memory_order_acquire)){
  //   if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation_inner() returning early because search is stopped";
  //   return;
  // }
  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Trying to get a lock on nodes reading for node: " << my_node->DebugString();
  search_->nodes_mutex_.lock_shared();
  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Got a lock on nodes reading for node: " << my_node->DebugString();

  // Unless this is the starting position, check what brought us here (for informational purposes)
  if(params_.GetAuxEngineVerbosity() >= 9 && search_->played_history_.GetLength() > 1){
    if(black_to_move){
      LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation_inner called with node" << my_node->DebugString() << " white to edge/move _to_ this node: " << my_node->GetOwnEdge()->GetMove(black_to_move).as_string() << " (debugging info for the edge: " << my_node->GetOwnEdge()->DebugString() << ") and this move from the a/b-helper: " << my_moves[ply].as_string() << "(seen from whites perspective) is really made by black, ply=" << ply;
    } else {
      LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation_inner called with node" << my_node->DebugString() << " black to edge/move _to_ this node: " << my_node->GetOwnEdge()->GetMove(black_to_move).as_string() << " and this move from the a/b-helper: " << my_moves[ply].as_string() << " is made by white, ply=" << ply;
    }
  }

  // If the current node is terminal it will not have any edges, and there is nothing more to do.
  if(my_node->IsTerminal()){
    // unlock nodes before returning.
    search_->nodes_mutex_.unlock_shared();
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Reached a terminal node, nothing to do. Releasing the lock on nodes";
    return;
  }

  // Find the edge
  for (auto& edge : my_node->Edges()) {
    if(edge.GetMove() == my_moves[ply] ){
      // Queue Leelas favourite node START
      // If there are children, find leelas preferred move, and if that move hasn't
      // already been queried, enqueue it, unless it is the same move as the helper suggests or depth is too high.
      int max_depth = 30;
      if(my_node->GetN() > 0 && ply < max_depth && amount_of_support > 0){
      	const EdgeAndNode Leelas_favourite = search_->GetBestChildNoTemperature(my_node, ply); // is this safe, or does it change my_node?
      	if(Leelas_favourite.edge() != edge.edge()){
      	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Leelas favourite move: " << Leelas_favourite.GetMove(black_to_move).as_string() << " is not the same has the helper recommendation " << edge.GetMove(black_to_move).as_string();
      	  if(Leelas_favourite.HasNode()){
	    // modify probability of best path if both nodes exists and has visits
	    if(Leelas_favourite.node()->GetN() > 0){
	      // Silently assume leelas favourite node is also node with best_q
	      probability_of_best_path = (1-(Leelas_favourite.node()->GetQ(0.0f) - my_node->GetQ(0.0f))) * probability_of_best_path;
	    }
      	    if(Leelas_favourite.node()->GetAuxEngineMove() == 0xffff){
      	      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Leelas favourite move has not been queried, it is " << Leelas_favourite.GetMove(black_to_move).as_string() << ", node: " << Leelas_favourite.DebugString() << ", queueing it now.";
      	      Node * n = Leelas_favourite.node();
      	      // Check that it's not terminal
      	      if(!n->IsTerminal()){
		// TODO: why is it needed to unlock here, I'd expected that the lock was necessary.		
      		search_->nodes_mutex_.unlock_shared();
      		AuxMaybeEnqueueNode(n);
      		search_->nodes_mutex_.lock_shared();
      	      } else {
      		if(params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Leelas favourite move leads to a terminal node: " << n->DebugString();
      	      }
      	    } else {
      	      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Leelas favourite move has already been queried. It is " << Leelas_favourite.GetMove(black_to_move).as_string() << ", node: " << Leelas_favourite.DebugString();
      	    }
      	  }
      	} else {
      	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Leelas favourite move is the same as the move recommmended by the helper. Nothing to do.";
	}
      }
      // Queue Leelas favourite node STOP
        
      edge_found = true;
      // If the edge is already extended, then just recursively call PreExtendTreeAndFastTrackForNNEvaluation_inner() with this node and ply increased by one.
      if(edge.HasNode()){
	if((int) my_moves.size() > ply+1){
	  if (params_.GetAuxEngineVerbosity() >= 9) {
	    if(black_to_move){
	      LOGFILE << "Blacks move " << edge.GetMove(black_to_move).as_string() << " (from white: " << edge.GetMove().as_string() << ") is expanded and has policy " << edge.GetP() << ". Go deeper.";
	    } else {
	      LOGFILE << "Whites move " << edge.GetMove(black_to_move).as_string() << " is expanded and has policy " << edge.GetP() << ". Go deeper.";	    
	    }
	  }
	  // unlock nodes so that the next level can write stuff.
	  search_->nodes_mutex_.unlock_shared();
	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Releasing lock before calling PreExtendTreeAndFastTrackForNNEvaluation_inner() recursively.";
	  PreExtendTreeAndFastTrackForNNEvaluation_inner(edge.node(), my_moves, ply+1, nodes_added, source, nodes_from_helper_added_by_this_PV, amount_of_support, probability_of_best_path);

	} else {
	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "All moves in the PV already expanded, nothing to do.";
	  // unlock nodes before returning.
	  search_->nodes_mutex_.unlock_shared();
	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Releasing lock before returning from PreExtendTreeAndFastTrackForNNEvaluation_inner()";
	  return;
	}
      } else {
	if (params_.GetAuxEngineVerbosity() >= 9){
	  if(black_to_move){	
	    LOGFILE << "Blacks move (edge) " << edge.GetMove(black_to_move).as_string() << " (from white: " << edge.GetMove().as_string() << ") is not expanded. It has policy " << edge.GetP() << ". Will expand it, and add the resulting node to the minibatch_, and then use it as parent";
	  } else {
	    LOGFILE << "Whites move (edge) " << edge.GetMove(black_to_move).as_string() << " is not expanded. It has policy " << edge.GetP() << " Will expand it, and add the resulting node to the minibatch_, and then use it as parent.";
	  }
	}

	search_->nodes_mutex_.unlock_shared();

	// Create a history variable that will be filled by the four argument version of ExtendNode().
	lczero::PositionHistory history = search_->played_history_;
	// copy the part of my_moves that makes up the history of this node
	std::vector<lczero::Move> moves_to_this_node;
	std::copy_n(my_moves.begin(), ply+1, std::back_inserter(moves_to_this_node));
	
	// Get a unique lock since GetOrSpawnNode() writes to the parent.
	// This can take a long time, don't do it if search is stopped.
	// Check if search is stopped.
	if(search_->stop_.load(std::memory_order_acquire)){
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation_inner() returning early because search is stopped";
	  return;
	}
	
	search_->nodes_mutex_.lock();
	// GetOrSpawnNode() does work with the lock on since it does not modify the tree.
	Node* child_node = edge.GetOrSpawnNode(my_node);
	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Node spawned";
	search_->nodes_mutex_.unlock();
	ExtendNode(child_node, ply+1, moves_to_this_node, &history); // This will modify history which will be re-used later here.
	// queue for NN evaluation.
	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Adding newly extended node: " << child_node->DebugString() << " to the minibatch_";

	// bool is_terminal=child_node->IsTerminal(); // while we have the lock.

	nodes_from_helper_added_by_this_PV->push_back(child_node); // shouldn't need a mutex.
	nodes_added++;

	// This part mostly copy/pasted from ProcessPickedTask
	if (!child_node->IsTerminal()) {
	  minibatch_.push_back(NodeToProcess::Visit(child_node, static_cast<uint16_t>(ply+1)));
	  minibatch_[minibatch_.size()-1].nn_queried = true;
	  // We adjust visit_in_flight in the node manually later when we know the actual number of new nodes added.
	  // FinalizeScoreUpdate() adds multivisit + n_in_flight, so having a non-zero multivisit would result in doubled N after backup.
	  minibatch_[minibatch_.size()-1].multivisit = 1;
	  // For now, do not re-implement the full cache-machinery in GatherMinibatch2() here.
	  // Accept a small inefficiency by not using the NNCache.
	  // For NN evaluation three things are needed: hash, input_planes and probabilities_to_cache.
	  minibatch_[minibatch_.size()-1].best_path_probability = probability_of_best_path;
	  const auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
	  minibatch_[minibatch_.size()-1].hash = hash;
	  int transform;
	  minibatch_[minibatch_.size()-1].input_planes = EncodePositionForNN(
		   search_->network_->GetCapabilities().input_format, history, 8,
		   params_.GetHistoryFill(), &transform);
	  minibatch_[minibatch_.size()-1].probability_transform = transform;
	  std::vector<uint16_t>& moves = minibatch_[minibatch_.size()-1].probabilities_to_cache;
	  // Legal moves are known, use them.
	  moves.reserve(child_node->GetNumEdges());
	  for (const auto& edge : child_node->Edges()) {
	    moves.emplace_back(edge.GetMove().as_nn_index(transform));
	  }
	  // Add the data to computation too, since GatherMinibatch2() will not touch our NodesToProcess
	  computation_->AddInput(minibatch_[minibatch_.size()-1].hash,
				 std::move(minibatch_[minibatch_.size()-1].input_planes),
				 std::move(minibatch_[minibatch_.size()-1].probabilities_to_cache));

	} else {
	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "The newly extended node is terminal";
	  minibatch_.push_back(NodeToProcess::Visit(child_node, 1)); // Only one visit, since this is a terminal
	  minibatch_[minibatch_.size()-1].nn_queried = false;
	  minibatch_[minibatch_.size()-1].ooo_completed = false;
	  child_node->IncrementNInFlight(1); // seems necessary.
	}

	// search_->nodes_mutex_.unlock();
	// // unlock the readlock.
	// search_->nodes_mutex_.unlock_shared();
	// if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Releasing lock on nodes. finished adding the node.";
	// if (!is_terminal && (int) my_moves.size() > ply+1){  
	if (!child_node->IsTerminal() && (int) my_moves.size() > ply+1){
	  // Go deeper.
	  PreExtendTreeAndFastTrackForNNEvaluation_inner(child_node, my_moves, ply+1, nodes_added, source, nodes_from_helper_added_by_this_PV, amount_of_support, probability_of_best_path);
	  return; // someone further down has already added visits_in_flight;
	}
	
	search_->search_stats_->pure_stats_mutex_.lock();
	search_->search_stats_->Number_of_nodes_added_by_AuxEngine += nodes_added;
	search_->search_stats_->pure_stats_mutex_.unlock();
	
	// Not going deeper now, either because the PV is finished, or because we hit a terminal node.
	// Aquire a write lock to adjust visits_in_flight.

	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Successfully added a full PV, depth = " << ply << ", number of nodes added = " << nodes_added;
	// Adjust the visits_in_flight now that we know how many nodes where actually added.
	// Each node shall have as many visits_in_flight as there are added nodes beneath it.
	// Increase the number of visits_in_flight to add for each generational step, until the
	// number is equal to nodes_added.
	int visits_to_add = 1;
	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Trying to aquire a lock on nodes to adjust IncrementNInFlight";	
	search_->nodes_mutex_.lock();
	for(Node * n = child_node; n != search_->root_node_; n = n->GetParent()){
	  n->IncrementNInFlight(visits_to_add);  
	  if(visits_to_add < nodes_added){
	    visits_to_add++;
	  }
	}
	// The loop above stops just before root, so fix root too. // TODO fix this ugly off-by-one hack. (perhaps test for n != nullptr)
	search_->root_node_->IncrementNInFlight(visits_to_add);
	search_->nodes_mutex_.unlock();
	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Unlocked the lock on nodes used to adjust IncrementNInFlight.";
	return;
      }
    }
  }
  if(!edge_found){
    // show full my_moves
    std::string s;
    PositionHistory ph = search_->played_history_;
    for(int i = 0; i <= ply; i++){
      LOGFILE << "debugging: ply = " << i << " " << GetFen(ph.Last());
      ph.Append(my_moves[i]);
      s = s + my_moves[i].as_string() + " ";
    }
    LOGFILE << "Debugging: ply=" << ply << " my_moves: " << s << " fen for root position: " << GetFen(search_->played_history_.Last());
    // ply is depth from root, so could be used to determine side to move.
    bool black_to_move = ! search_->played_history_.IsBlackToMove() ^ (ply % 2 == 0);
    Move m;
    if (!Move::ParseMove(&m, my_moves[ply].as_string(), black_to_move)) {
      LOGFILE << "Bad move: " << my_moves[ply].as_string() << black_to_move;
    } else {
      LOGFILE << "edge not found! from white side:" << my_moves[ply].as_string() << " absolute: " << m.as_string();
      // perhaps Leela marked this a "terminal" due to repetitions, if so, then there are no edges.
      for (auto& edge : my_node->Edges()) {
      	LOGFILE << "Edge " << edge.GetMove().as_string() << " does not match the move from the A/B helper: " << my_moves[ply].as_string() ;
	edge_found = true;
      }
    }
    if(edge_found){
      LOGFILE << "Leelas node has edges, but the recommended move was not found among them! Is this a case of https://github.com/hans-ekbrand/lc0/issues/2 where we mistakenly belive a move is castling when it isn't?";
      throw Exception("Leelas node has edges, but the recommended move was not found among them!");
    } else {
      LOGFILE << "No edges found, repetition?";
    }
    // Release the read lock before returning
    search_->nodes_mutex_.unlock_shared();
    if (params_.GetAuxEngineVerbosity() >= 3) LOGFILE << "Released the lock on nodes in debugging.";    
  }
}

// 1.5 Extend tree with nodes using PV of a/b helper, and add the new nodes to the minibatch.

const std::shared_ptr<Search::adjust_policy_stats> SearchWorker::PreExtendTreeAndFastTrackForNNEvaluation() {
// std::queue<std::vector<Node*>> SearchWorker::PreExtendTreeAndFastTrackForNNEvaluation() {
  // input: a PV starting from root in form of a vector of Moves (the vectors are stored in a global queue called fast_track_extend_and_evaluate_queue_)
  // LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation() for thread: " << std::hash<std::thread::id>{}(std::this_thread::get_id());
  // lock the queue before reading from it
  // Check if search is stopped before trying to take a lock

  // Never add more than 256 nodes to the batch (or you may get these errrors: exception.h:39] Exception: CUDNN error: CUDNN_STATUS_BAD_PARAM (../../src/neural/cuda/layers.cc:228) if max_batch=512
  // options.GetOrDefault<int>("max_batch", 1024)

  // Also never add more than 20 PV:s per batch, or search risks getting stuck adding nodes instead of evaluating them.

  std::queue<std::vector<Node*>> queue_of_vector_of_nodes_from_helper_added_by_this_thread = {};
  const std::shared_ptr<Search::adjust_policy_stats> bar = std::make_unique<Search::adjust_policy_stats>();

  // Check if search_stats_->initial_purge_run == true. If it is not, then return early, because than AuxWorker() thread 0 hasn't purged the PV:s yet.
  // to read search_stats_->initial_purge_run, take a lock on pure_stats_.
  
  search_->search_stats_->pure_stats_mutex_.lock_shared();
  if(!search_->search_stats_->initial_purge_run){
    search_->search_stats_->pure_stats_mutex_.unlock_shared();
    return(bar);
  }
  search_->search_stats_->pure_stats_mutex_.unlock_shared();
      
  search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock(); // lock this queue before reading from it.
  if(search_->search_stats_->fast_track_extend_and_evaluate_queue_.size() > 0){

    search_->search_stats_->pure_stats_mutex_.lock_shared();
    int number_of_added_nodes_at_start = search_->search_stats_->Number_of_nodes_added_by_AuxEngine;
    int number_of_PVs_added = 0;
    
    while(search_->search_stats_->fast_track_extend_and_evaluate_queue_.size() > 0 &&
	  search_->search_stats_->Number_of_nodes_added_by_AuxEngine - number_of_added_nodes_at_start < static_cast<long long unsigned int>(std::floor(params_.GetMiniBatchSize() * 0.3)) && 
	  number_of_PVs_added < static_cast<int>(std::floor(params_.GetMiniBatchSize() * 0.3)) // don't drag the speed down.
	  ){
      // relase the lock, we only needed it to test if to continue or not
      search_->search_stats_->pure_stats_mutex_.unlock_shared();

      if (params_.GetAuxEngineVerbosity() >= 5) {
	LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation: size of minibatch_ is " << minibatch_.size();
	LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation: size of search_stats_->fast_track_extend_and_evaluate_queue_ is " << search_->search_stats_->fast_track_extend_and_evaluate_queue_.size();
      }
      std::vector<lczero::Move> my_moves = search_->search_stats_->fast_track_extend_and_evaluate_queue_.front(); // read the element
      search_->search_stats_->fast_track_extend_and_evaluate_queue_.pop(); // remove it from the queue.
      int amount_of_support_for_PVs_ = search_->search_stats_->amount_of_support_for_PVs_.front();
      // LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation: size of amount_of_support_for_PVs_ is " << search_->search_stats_->amount_of_support_for_PVs_.size();
      search_->search_stats_->amount_of_support_for_PVs_.pop();
      int starting_depth_of_PVs_ = search_->search_stats_->starting_depth_of_PVs_.front();
      search_->search_stats_->starting_depth_of_PVs_.pop();
      // LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation: size of starting_depth_of_PVs_ is " << search_->search_stats_->amount_of_support_for_PVs_.size();      
      // int source = search_->search_stats_->source_of_PVs.front();
      // search_->search_stats_->source_of_PVs.pop();
      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation() popped the node queue (current size: " << search_->search_stats_->fast_track_extend_and_evaluate_queue_.size() << ").";
      // Finished modifying the queue, release the lock, so that others can add more PVs to it while we extend nodes.
      // long unsigned int queue_size = search_->search_stats_->fast_track_extend_and_evaluate_queue_.size();
      search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock(); // unlock
      
      if (params_.GetAuxEngineVerbosity() >= 9) {	
	// show full my_moves
	std::string s;
	if(my_moves.size() > 0){
	  for(int i = 0; i < (int) my_moves.size(); i++){
	    s = s + my_moves[i].as_string() + " ";
	  }
	  LOGFILE << "Length of PV to add: " << my_moves.size() << " my_moves: " << s << ".";
	}
      }
      int source = 0; // dummy while we don't track source for the moment.
      std::vector<Node*> nodes_from_helper_added_by_this_PV = {};
      // LOGFILE << "size: " << nodes_from_helper_added_by_this_PV.size();
      PreExtendTreeAndFastTrackForNNEvaluation_inner(search_->root_node_, my_moves, 0, 0, source, &nodes_from_helper_added_by_this_PV, amount_of_support_for_PVs_, 1.0f);
      number_of_PVs_added++;
      if (nodes_from_helper_added_by_this_PV.size() > 0){
	// add this vector to the queue, since it is not empty
	// queue_of_vector_of_nodes_from_helper_added_by_this_thread.push(nodes_from_helper_added_by_this_PV);
	bar->queue_of_vector_of_nodes_from_helper_added_by_this_thread.push(nodes_from_helper_added_by_this_PV);
	bar->amount_of_support_for_PVs_.push(amount_of_support_for_PVs_);
	bar->starting_depth_of_PVs_.push(starting_depth_of_PVs_);
      }
      if (params_.GetAuxEngineVerbosity() >= 9) {
	std::thread::id this_id = std::this_thread::get_id();
	LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation: finished one iteration, size of minibatch_ is " << minibatch_.size();
	LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation: finished one iteration, size of nodes_from_helper_added_by_this_PV is " << nodes_from_helper_added_by_this_PV.size();
	LOGFILE << "Thread: " << this_id << ", PreExtendTreeAndFastTrackForNNEvaluation: finished one iteration, size of queue_of_vector_of_nodes_from_helper_added_by_this_thread is " << queue_of_vector_of_nodes_from_helper_added_by_this_thread.size();
	LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation: finished one iteration, size of search_stats_->fast_track_extend_and_evaluate_queue_ is " << search_->search_stats_->fast_track_extend_and_evaluate_queue_.size();
      }

      // Check if search is stopped.
      if(search_->stop_.load(std::memory_order_acquire)){
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation() returning early because search is stopped";
	return(bar);
      }

      // While we extended nodes, someone could have added more PV:s, update our belief about the current size of the queue.
      search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock(); // lock this queue before reading from it again.
      search_->search_stats_->pure_stats_mutex_.lock_shared(); // Always end the while loop with the lock on.
    } // end of while loop
    // If there are nodes left, then fill them with nodes along the highest probability path.    
    number_of_added_nodes_at_start = search_->search_stats_->Number_of_nodes_added_by_AuxEngine;
    search_->search_stats_->pure_stats_mutex_.unlock_shared();
    // Under construction:
    // 1. make root the current node
    // 2. make the best child of the current node the current node
    // 3. if current node has any unextended nodes, extend them, by adding at most the nodes left to add.
    // 4. if there are nodes to add left, then goto 2.
    
  }
  search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock(); // unlock
  // LOGFILE << "PreExtendTreeAndFastTrackForNNEvaluation: finished.";
  return(bar);
}
  
// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
namespace {
int Mix(int high, int low, float ratio) {
  return static_cast<int>(std::round(static_cast<float>(low) +
                                     static_cast<float>(high - low) * ratio));
}

int CalculateCollisionsLeft(int64_t nodes, const SearchParams& params) {
  // End checked first
  if (nodes >= params.GetMaxCollisionVisitsScalingEnd()) {
    return params.GetMaxCollisionVisits();
  }
  if (nodes <= params.GetMaxCollisionVisitsScalingStart()) {
    return 1;
  }
  return Mix(params.GetMaxCollisionVisits(), 1,
             std::pow((static_cast<float>(nodes) -
                       params.GetMaxCollisionVisitsScalingStart()) /
                          (params.GetMaxCollisionVisitsScalingEnd() -
                           params.GetMaxCollisionVisitsScalingStart()),
                      params.GetMaxCollisionVisitsScalingPower()));
}
}  // namespace

void SearchWorker::GatherMinibatch2(int number_of_nodes_already_added) {
  if (params_.GetAuxEngineVerbosity() >= 2 && number_of_nodes_already_added > 0) LOGFILE << "GatherMinibatch2() called with " << number_of_nodes_already_added;
  // Total number of nodes to process.
  int minibatch_size = 0;
  int cur_n = 0;
  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << "GatherMinibatch2() trying to aquire a shared lock on nodes";  
  {
    SharedMutex::Lock lock(search_->nodes_mutex_);
    cur_n = search_->root_node_->GetN();
  }
  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << "GatherMinibatch2() Aquired and released a shared lock on nodes";

  // const double remaining_time = latest_time_manager_hints_.GetEstimatedRemainingTimeMs() / 1000.0;
  // LOGFILE << "Remaining time: " << remaining_time;
  
  // TODO: GetEstimatedRemainingPlayouts has already had smart pruning factor
  // applied, which doesn't clearly make sense to include here...
  int64_t remaining_n =
      latest_time_manager_hints_.GetEstimatedRemainingPlayouts();
  int collisions_left = CalculateCollisionsLeft(
      std::min(static_cast<int64_t>(cur_n), remaining_n), params_);

  // Number of nodes processed out of order.
  number_out_of_order_ = 0;

  int thread_count = search_->thread_count_.load(std::memory_order_acquire);

  int iteration_counter = 0;

  // Gather nodes to process in the current batch.
  // If we had too many nodes out of order, also interrupt the iteration so
  // that search can exit.

  while (minibatch_size < params_.GetMiniBatchSize() - number_of_nodes_already_added &&
         number_out_of_order_ < params_.GetMaxOutOfOrderEvals()) {
    // If there's something to process without touching slow neural net, do it.
    if (minibatch_size > 0 && computation_->GetCacheMisses() == 0) return;

    // If there is backend work to be done, and the backend is idle - exit
    // immediately.
    // Only do this fancy work if there are multiple threads as otherwise we
    // early exit from every batch since there is never another search thread to
    // be keeping the backend busy. Which would mean that threads=1 has a
    // massive nps drop.
    if (thread_count > 1 && minibatch_size > static_cast<int>(std::floor(params_.GetMiniBatchSize() * 0.6)) &&
        computation_->GetCacheMisses() > params_.GetIdlingMinimumWork() &&
        thread_count - search_->backend_waiting_counter_.load(
                           std::memory_order_relaxed) >
            params_.GetThreadIdlingThreshold()) {
      return;
    }

    int new_start = static_cast<int>(minibatch_.size());

    if(iteration_counter < 10){
      // First k runs are custom runs which may override CPUCT and force visits into a specific line.
      int max_force_visits = std::min({collisions_left, params_.GetMiniBatchSize() - number_of_nodes_already_added - minibatch_size,
	  params_.GetMaxOutOfOrderEvals() - number_out_of_order_});
      PickNodesToExtend(max_force_visits, iteration_counter+1);
    } else {
      // Normal run
      PickNodesToExtend(
			  std::min({collisions_left, params_.GetMiniBatchSize() - number_of_nodes_already_added - minibatch_size,
			      params_.GetMaxOutOfOrderEvals() - number_out_of_order_}), 0);
    }
    iteration_counter++;

      // Count the non-collisions.
      int non_collisions = 0;
      for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
	auto& picked_node = minibatch_[i];
	if (picked_node.IsCollision()) {
	  continue;
	}
	++non_collisions;
	++minibatch_size;
      }

    bool needs_wait = false;
    int ppt_start = new_start;
    if (params_.GetTaskWorkersPerSearchWorker() > 0 &&
        non_collisions >= params_.GetMinimumWorkSizeForProcessing()) {
      const int num_tasks = std::clamp(
          non_collisions / params_.GetMinimumWorkPerTaskForProcessing(), 2,
          params_.GetTaskWorkersPerSearchWorker() + 1);
      // Round down, left overs can go to main thread so it waits less.
      int per_worker = non_collisions / num_tasks;
      needs_wait = true;
      // LOGFILE << "SearchWorker::GatherMinibatch2() calling ResetTasks() for thread: " << std::hash<std::thread::id>{}(std::this_thread::get_id());
      ResetTasks();
      int found = 0;
      for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
        auto& picked_node = minibatch_[i];
        if (picked_node.IsCollision()) {
          continue;
        }
        ++found;
        if (found == per_worker) {
          picking_tasks_.emplace_back(ppt_start, i + 1);
          task_count_.fetch_add(1, std::memory_order_acq_rel);
          ppt_start = i + 1;
          found = 0;
          if (picking_tasks_.size() == static_cast<size_t>(num_tasks - 1)) {
            break;
          }
        }
      }
    }
    ProcessPickedTask(ppt_start, static_cast<int>(minibatch_.size()),
                      &main_workspace_);
    if (needs_wait) {
      WaitForTasks();
    }
    bool some_ooo = false;
    for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) {
      if (minibatch_[i].ooo_completed) {
        some_ooo = true;
        break;
      }
    }
    if (some_ooo) {
      SharedMutex::Lock lock(search_->nodes_mutex_);
      for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start;
           i--) {
        // If there was any OOO, revert 'all' new collisions - it isn't possible
        // to identify exactly which ones are afterwards and only prune those.
        // This may remove too many items, but hopefully most of the time they
        // will just be added back in the same in the next gather.
        if (minibatch_[i].IsCollision()) {
          Node* node = minibatch_[i].node;
          for (node = node->GetParent();
               node != search_->root_node_->GetParent();
               node = node->GetParent()) {
            node->CancelScoreUpdate(minibatch_[i].multivisit);
          }
          minibatch_.erase(minibatch_.begin() + i);
        } else if (minibatch_[i].ooo_completed) {
          DoBackupUpdateSingleNode(minibatch_[i]);
          minibatch_.erase(minibatch_.begin() + i);
          --minibatch_size;
          ++number_out_of_order_;
        }
      }
      if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << "GatherMinibatch2() Aquired and released a shared lock on nodes 2.";      
    }
    for (size_t i = new_start; i < minibatch_.size(); i++) {
      // If there was no OOO, there can stil be collisions.
      // There are no OOO though.
      // Also terminals when OOO is disabled.
      if (!minibatch_[i].nn_queried) continue;
      if (minibatch_[i].is_cache_hit) {
        // Since minibatch_[i] holds cache lock, this is guaranteed to succeed.
        computation_->AddInputByHash(minibatch_[i].hash,
                                     std::move(minibatch_[i].lock));
      } else {
        computation_->AddInput(minibatch_[i].hash,
                               std::move(minibatch_[i].input_planes),
                               std::move(minibatch_[i].probabilities_to_cache));
      }
    }

    // Check for stop at the end so we have at least one node.
    for (size_t i = new_start; i < minibatch_.size(); i++) {
      auto& picked_node = minibatch_[i];

      if (picked_node.IsCollision()) {
        // Check to see if we can upsize the collision to exit sooner.
        if (picked_node.maxvisit > 0 &&
            collisions_left > picked_node.multivisit) {
          SharedMutex::Lock lock(search_->nodes_mutex_);
          int extra = std::min(picked_node.maxvisit, collisions_left) -
                      picked_node.multivisit;
          picked_node.multivisit += extra;
          Node* node = picked_node.node;
          for (node = node->GetParent();
               node != search_->root_node_->GetParent();
               node = node->GetParent()) {
            node->IncrementNInFlight(extra);
          }
	  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << "GatherMinibatch2() Aquired and released a shared lock on nodes 3.";	  
        }
        if ((collisions_left -= picked_node.multivisit) <= 0) return;
        if (search_->stop_.load(std::memory_order_acquire)) return;
      }
    }
    // if (params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "GatherMinibatch2() iteration counter: " << iteration_counter << " collisions: " << stats_non_collisions << " out of " << minibatch_.size() - number_of_nodes_already_added << " tested nodes";

  }
}

void SearchWorker::ProcessPickedTask(int start_idx, int end_idx,
                                     TaskWorkspace* workspace) {
  auto& history = workspace->history;
  history = search_->played_history_;
  // LOGFILE << "ProcessPickedTask starting at node " << start_idx;

  for (int i = start_idx; i < end_idx; i++) {
    auto& picked_node = minibatch_[i];
    if (picked_node.IsCollision()) continue;
    auto* node = picked_node.node;

    // If node is already known as terminal (win/loss/draw according to rules
    // of the game), it means that we already visited this node before.
    if (picked_node.IsExtendable()) {
      // LOGFILE << "ProcessPickedTask at node " << i << " " << node->DebugString() << " is extendable";
      
      // Node was never visited, extend it.
      ExtendNode(node, picked_node.depth, picked_node.moves_to_visit, &history);
      if (!node->IsTerminal()) {
        picked_node.nn_queried = true;
        const auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
        picked_node.hash = hash;
        picked_node.lock = NNCacheLock(search_->cache_, hash);
        picked_node.is_cache_hit = picked_node.lock;
        if (!picked_node.is_cache_hit) {
          int transform;
          picked_node.input_planes = EncodePositionForNN(
              search_->network_->GetCapabilities().input_format, history, 8,
              params_.GetHistoryFill(), &transform);
          picked_node.probability_transform = transform;

          std::vector<uint16_t>& moves = picked_node.probabilities_to_cache;
          // Legal moves are known, use them.
          moves.reserve(node->GetNumEdges());
          for (const auto& edge : node->Edges()) {
            moves.emplace_back(edge.GetMove().as_nn_index(transform));
          }
        } else {
          picked_node.probability_transform = TransformForPosition(
              search_->network_->GetCapabilities().input_format, history);
        }
      }
    }
    if (params_.GetOutOfOrderEval() && picked_node.CanEvalOutOfOrder()) {
      // Perform out of order eval for the last entry in minibatch_.
      FetchSingleNodeResult(&picked_node, picked_node, 0);
      picked_node.ooo_completed = true;
    }
  }
}

#define MAX_TASKS 100

void SearchWorker::ResetTasks() {
  task_count_.store(0, std::memory_order_release);
  tasks_taken_.store(0, std::memory_order_release);
  completed_tasks_.store(0, std::memory_order_release);
  picking_tasks_.clear();
  // Reserve because resizing breaks pointers held by the task threads.
  picking_tasks_.reserve(MAX_TASKS);
}

int SearchWorker::WaitForTasks() {
  // Spin lock, other tasks should be done soon.
  while (true) {
    int completed = completed_tasks_.load(std::memory_order_acquire);
    int todo = task_count_.load(std::memory_order_acquire);
    if (todo == completed) return completed;
    SpinloopPause();
  }
}

void SearchWorker::PickNodesToExtend(int collision_limit, int override_cpuct) {
  ResetTasks();
  {
    // While nothing is ready yet - wake the task runners so they are ready to
    // receive quickly.
    Mutex::Lock lock(picking_tasks_mutex_);
    task_added_.notify_all();
  }
  std::vector<Move> empty_movelist;
  // This lock must be held until after the task_completed_ wait succeeds below.
  // Since the tasks perform work which assumes they have the lock, even though
  // actually this thread does.
  SharedMutex::Lock lock(search_->nodes_mutex_);
  bool more_work = PickNodesToExtendTask(search_->root_node_, 0, collision_limit, empty_movelist,
                        &minibatch_, &main_workspace_, 1, 0, override_cpuct);
  if(more_work){
    WaitForTasks();
    for (int i = 0; i < static_cast<int>(picking_tasks_.size()); i++) {
      for (int j = 0; j < static_cast<int>(picking_tasks_[i].results.size());
	   j++) {
	minibatch_.emplace_back(std::move(picking_tasks_[i].results[j]));
      }
    }
  }
}

void SearchWorker::EnsureNodeTwoFoldCorrectForDepth(Node* child_node,
                                                    int depth) {
  // Check whether first repetition was before root. If yes, remove
  // terminal status of node and revert all visits in the tree.
  // Length of repetition was stored in m_. This code will only do
  // something when tree is reused and twofold visits need to be
  // reverted.
  if (child_node->IsTwoFoldTerminal() && depth < child_node->GetM()) {
    // Take a mutex - any SearchWorker specific mutex... since this is
    // not safe to do concurrently between multiple tasks.
    Mutex::Lock lock(picking_tasks_mutex_);
    int depth_counter = 0;
    // Cache node's values as we reset them in the process. We could
    // manually set wl and d, but if we want to reuse this for reverting
    // other terminal nodes this is the way to go.
    const auto wl = child_node->GetWL();
    const auto d = child_node->GetD();
    const auto m = child_node->GetM();
    const auto terminal_visits = child_node->GetN();
    for (Node* node_to_revert = child_node; node_to_revert != nullptr;
         node_to_revert = node_to_revert->GetParent()) {
      // Revert all visits on twofold draw when making it non terminal.
      node_to_revert->RevertTerminalVisits(wl, d, m + (float)depth_counter,
                                           terminal_visits);
      depth_counter++;
      // Even if original tree still exists, we don't want to revert
      // more than until new root.
      if (depth_counter > depth) break;
      // If wl != 0, we would have to switch signs at each depth.
    }
    // Mark the prior twofold draw as non terminal to extend it again.
    child_node->MakeNotTerminal();
    // When reverting the visits, we also need to revert the initial
    // visits, as we reused fewer nodes than anticipated.
    search_->initial_visits_ -= terminal_visits;
    // Max depth doesn't change when reverting the visits, and
    // cum_depth_ only counts the average depth of new nodes, not reused
    // ones.
  }
}

bool SearchWorker::PickNodesToExtendTask(Node* node, int base_depth,
                                         int collision_limit,
                                         const std::vector<Move>& moves_to_base,
                                         std::vector<NodeToProcess>* receiver,
                                         TaskWorkspace* workspace,
					 float probability_of_best_path,
					 int distance_from_best_path,
					 int override_cpuct) {

  // TODO: Bring back pre-cached nodes created outside locks in a way that works
  // with tasks.
  // TODO: pre-reserve visits_to_perform for expected depth and likely maximum
  // width. Maybe even do so outside of lock scope.

  int override_cpuct_number_of_times = 10; // only test if cpuct should be overriden the first k times when the function is called

  if(override_cpuct > 0 && override_cpuct < override_cpuct_number_of_times){

    // There are two different types of boosting to make:
    // 1. Starting points of divergencies (force more visits to these)
       // A. The helper's prefererred line
    // 2. Successively deepened entry points (force less visits to these).
      // A. An interesting node somewhere in 1A
    // When/if the node to be boosted has as many visits as its competing sibling, stop boosting it, since Leela uses number of visits to decide "best child".

    int collision_limit_one; // the amount of visits we decide to use

    search_->search_stats_->best_move_candidates_mutex.lock_shared(); // for _reading_ search_stats_->winning_ and others
    search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_.lock(); // for reading Helpers_preferred_child_node_ and vector_of_moves_from_root_to_Helpers_preferred_child_node_ and the other two.
    
    // if(search_->search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child > 0 &&
    //    search_->search_stats_->Helpers_preferred_child_node_in_Leelas_PV_ != nullptr &&
    //    search_->search_stats_->Helpers_preferred_child_node_ != nullptr &&
    //    search_->search_stats_->Helpers_preferred_child_node_->GetN() > 0 &&
    //    search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_.size() > 0 &&
    //    search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_in_Leelas_PV_.size() > 0
    //    ){
    if(search_->search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child > 0 &&
       search_->search_stats_->number_of_nodes_in_support_for_helper_eval_of_helpers_preferred_child > 0 &&
       search_->search_stats_->Helpers_preferred_child_node_in_Leelas_PV_ != nullptr &&
       search_->search_stats_->Helpers_preferred_child_node_ != nullptr &&
       search_->search_stats_->Helpers_preferred_child_node_->GetN() > 0 &&
       search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_.size() > 0 &&
       search_->search_stats_->helper_PV_from_instance_two_explore_node != nullptr &&
       search_->search_stats_->helper_PV_from_instance_two_explore_moves.size() > 0
       ){

      int centipawn_diff = std::abs(search_->search_stats_->helper_eval_of_leelas_preferred_child - search_->search_stats_->helper_eval_of_helpers_preferred_child);
      if(override_cpuct == 1){
	LOGFILE << "Centipawn diff: " << centipawn_diff << " leelas_preferred_child: " << search_->search_stats_->helper_eval_of_leelas_preferred_child << " helpers_preferred_child: " << search_->search_stats_->helper_eval_of_helpers_preferred_child << " divergence at depth: " << search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_.size() - 1 ;
      }

      bool donate_visits = false; // if the helper thinks leelas line is better, donate the visits to the parent which will boost visits to the point in the tree where they agree.
      bool roughly_equal = false; // if false then the helper's recommendation is either clearly worse or clearly better.

      if(centipawn_diff < 10){
	// helper is about equal, case B
	if(override_cpuct == 1 && params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "SearchWorker::PickNodesToExtendTask() Case B helper eval of helper preferred line is about equal (" << search_->search_stats_->helper_eval_of_helpers_preferred_child << ") to helper eval of Leelas PV (" << search_->search_stats_->helper_eval_of_leelas_preferred_child << ").";
	collision_limit_one = std::max(0, static_cast<int>(std::floor(collision_limit * params_.GetAuxEngineForceVisitsRatio())));
	roughly_equal = true;
      } else {
	if((search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_.size() % 2) != 0){ // an odd size implies an even distance!
	  if(search_->search_stats_->helper_eval_of_leelas_preferred_child < search_->search_stats_->helper_eval_of_helpers_preferred_child){
	    // helper is 'clearly' better
	    collision_limit_one = std::max(0, static_cast<int>(std::floor(collision_limit * 1 / 3.0f)));	    
	    if(override_cpuct == 1 && params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "SearchWorker::PickNodesToExtendTask() Case A1 found divergence at an even distance from root, so maximising helper eval, and helper eval of helper preferred line is higher (" << search_->search_stats_->helper_eval_of_helpers_preferred_child << ") than helper eval of Leelas PV (" << search_->search_stats_->helper_eval_of_leelas_preferred_child << "). This means the helper has found a better move for the side to move (Leela).";
	  } else {
	    // Helper is 'clearly' worse, case C
	    if(override_cpuct == 1 && params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "SearchWorker::PickNodesToExtendTask() Case C1 found divergence at an even distance from root, so maximising helper eval, and helper eval of helper preferred line is lower (" << search_->search_stats_->helper_eval_of_helpers_preferred_child << ") than helper eval of Leelas PV (" << search_->search_stats_->helper_eval_of_leelas_preferred_child << "), so Leela has found a better move for the side to move (Leela).";
	    donate_visits = true;
	    collision_limit_one = std::max(0, static_cast<int>(std::floor(collision_limit * params_.GetAuxEngineForceVisitsRatioInferiorLine())));
	  }
	} else {
	  // helper is 'clearly' better 
	  if(search_->search_stats_->helper_eval_of_leelas_preferred_child > search_->search_stats_->helper_eval_of_helpers_preferred_child){
	    collision_limit_one = std::max(0, static_cast<int>(std::floor(collision_limit * 1 / 3.0f)));
	    // collision_limit_one = std::max(0, static_cast<int>(std::floor(collision_limit * params_.GetAuxEngineForceVisitsRatio())));
	    if(override_cpuct == 1 && params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "SearchWorker::PickNodesToExtendTask() Case A2 found divergence at an odd distance from root, so minimising helper eval, and helper eval of helper preferred line is lower (" << search_->search_stats_->helper_eval_of_helpers_preferred_child << ") than helper eval of Leelas PV (" << search_->search_stats_->helper_eval_of_leelas_preferred_child << "). This means the helper has found a better move the for the opponent.";
	  } else {
	    // Case C, helper even worse than Leela +0.05 when minimising
	    if(override_cpuct == 1 && params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "SearchWorker::PickNodesToExtendTask() Case C2 found divergence at an odd distance from root, so minimising helper eval, and helper eval of helper preferred line is higher (" << search_->search_stats_->helper_eval_of_helpers_preferred_child << ") than helper eval of Leelas PV (" << search_->search_stats_->helper_eval_of_leelas_preferred_child << "), so Leela has found a better move for the opponent.";
	    donate_visits = true;
	    collision_limit_one = std::max(0, static_cast<int>(std::floor(collision_limit * params_.GetAuxEngineForceVisitsRatioInferiorLine())));
	  }
	}
      }

      // search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_.unlock();
      // fetch the helper_PV_from_instance_one_explore_moves and helper_PV_from_instance_two_explore_node before changing locks
      // std::vector<Move> helper_PV_from_instance_two_explore_moves = search_->search_stats_->helper_PV_from_instance_two_explore_moves;
      // Node* helper_PV_from_instance_two_explore_node = search_->search_stats_->helper_PV_from_instance_two_explore_node;
      // Instance one:
      
      // search_->search_stats_->best_move_candidates_mutex.unlock();
      // search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_.lock();

      Node* boosted_node;
      std::vector<Move> vector_of_moves_from_root_to_boosted_node;
      int hard_max_minibatch_size = 980; // 1024 if you have enough RAM on the GPU.

      // Need to define three things: (1) boosted_node, (2) vector_of_moves_from_root_to_boosted_node (3) collision_limit_one (i.e. the number of visits to force)
      boosted_node = search_->search_stats_->Helpers_preferred_child_node_;
      vector_of_moves_from_root_to_boosted_node = search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_;
      int divergence_at_depth = vector_of_moves_from_root_to_boosted_node.size();

      Node* best_child = search_->GetBestChildNoTemperature(boosted_node->GetParent(), vector_of_moves_from_root_to_boosted_node.size()).node();
      uint32_t visits_missing_to_be_equal = best_child->GetN() - boosted_node->GetN();
      uint32_t visits_missing_to_be_equal_including_in_flight = best_child->GetN() + best_child->GetNInFlight() - boosted_node->GetN() - boosted_node->GetNInFlight();
      collision_limit_one = std::max(static_cast<uint32_t>(collision_limit), hard_max_minibatch_size - boosted_node->GetNInFlight()); // Try to catch up fast.
      if(donate_visits){
	  // collision_limit_one = 0;
	  // // Do something useful with the information that both Leela and the helper agree up to the point of the second divergence. It should be rather low risk to boost that node, but no need to through ALL visits there.
	  // // Potentially, this node is very deep, and the game may never end up in that state. Perhaps it is useful to express the amount of boosting to spend there relative to that nodes current number of visits?
	  // // Say, std::min(std::max(2, 10% of its current visits), 50% of the available visits in this batch)?

	  // If this is root, then do not bother
	boosted_node = search_->search_stats_->Helpers_preferred_child_node_in_Leelas_PV_;
	// boosted_node is changed, also change the limit to match visits-in-flight of the boosted node.
	collision_limit_one = std::max(static_cast<uint32_t>(collision_limit), hard_max_minibatch_size - boosted_node->GetNInFlight()); // Try to catch up fast.
	if(boosted_node == search_->root_node_ || search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_in_Leelas_PV_.size() == 0){
	  collision_limit_one = 0;
	  if(boosted_node == search_->root_node_){
	    LOGFILE << "Anomaly detected: The second divergence is root node.";
	  } else {
	    LOGFILE << "Anomaly detected: The vector of moves up to the second divergence is of length zero.";
	  }
	} else {
	  // // Lets try something greedy!
	  // collision_limit_one = std::max(static_cast<uint32_t>(collision_limit), 1024 - boosted_node->GetNInFlight());
	  // // The greedy approach isn't too bad, -13 Elo, but perhaps it is better to only make sure that the helper's recommendation gets as many visits as the best child?

	  // Need to redefine best_child since we have changed boosted_node.
	  vector_of_moves_from_root_to_boosted_node = search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_in_Leelas_PV_;	    
	  best_child = search_->GetBestChildNoTemperature(boosted_node->GetParent(), vector_of_moves_from_root_to_boosted_node.size()).node();

	    // Copy code from `roughly_equal`
	  if(boosted_node == best_child || boosted_node->GetN() + boosted_node->GetNInFlight() >= best_child->GetN() + best_child->GetNInFlight()){
	    if(boosted_node == best_child){
	      LOGFILE << "Second divergence: already best child, stop boosting here.";
	    } else {
	      LOGFILE << "Second divergence: more visits but not yet best child, stop boosting here.";		
	    }
	    collision_limit_one = 0;
	  }
	  if(boosted_node->GetN() + boosted_node->GetNInFlight() + collision_limit_one > best_child->GetN() + best_child->GetNInFlight()){
	      // Equal number of visits is OK, but not more
	    if(boosted_node->GetN() + boosted_node->GetNInFlight() < best_child->GetN() + best_child->GetNInFlight()){
	      collision_limit_one = best_child->GetN() + best_child->GetNInFlight() - boosted_node->GetN() - boosted_node->GetNInFlight() - 1;
	      LOGFILE << "Second divergence: Limiting the number of forced visits to match best child.";
	    }
	  }
	}
	// Can we go to the end of this PV instead of the first node? For now we can at least to go "some interesting node".
	boosted_node = search_->search_stats_->helper_PV_from_instance_one_explore_node;
	vector_of_moves_from_root_to_boosted_node = search_->search_stats_->helper_PV_from_instance_one_explore_moves;
	// END OF DONATE VISITS
	
      } else {
	  // roughly equal or clearly better
	if(roughly_equal){
	    // don't boost the node if it is already best child,
	    // best_child is not always the child with highest N, when using QBasedMoveSelection
	  if(boosted_node == best_child || boosted_node->GetN() + boosted_node->GetNInFlight() >= best_child->GetN() + best_child->GetNInFlight()){
	    if(boosted_node == best_child){
	      LOGFILE << "Case 1: not clearly better, already best child, stop boosting here.";
	    } else {
	      LOGFILE << "Case 1: not clearly better, more visits but not yet best child, stop boosting here.";		
	    }
	    collision_limit_one = 0;
	  }
	  if(boosted_node->GetN() + boosted_node->GetNInFlight() + collision_limit_one > best_child->GetN() + best_child->GetNInFlight()){
	      // Equal number of visits is OK, but not more
	    if(boosted_node->GetN() + boosted_node->GetNInFlight() < best_child->GetN() + best_child->GetNInFlight()){
	      collision_limit_one = best_child->GetN() + best_child->GetNInFlight() - boosted_node->GetN() - boosted_node->GetNInFlight() - 1;
	      LOGFILE << "Case 1: not clearly better Limiting the number of forced visits to match best child.";
	    }
	  }
	} else {
	    // Clearly better,
	  if(boosted_node == best_child || boosted_node->GetN() + boosted_node->GetNInFlight() >= best_child->GetN() + best_child->GetNInFlight()){
	    // if(boosted_node->GetN() > best_child->GetN() + collision_limit_one){
	    // Continue to boost even when the node has more visits, but decrease the boost to let Leela have the final say.
	    if(boosted_node == best_child) {
	      LOGFILE << "Case 1: Clearly better, even best child now. Stop boosting now. Visits for best child "
		      << best_child->GetN() << " visits in flight for best child: " << best_child->GetNInFlight()
		      << " visits for boosted node: " << boosted_node->GetN() << " visits in flight for boosted node: "
		      << boosted_node->GetNInFlight();
	    } else {
	      LOGFILE << "Case 1: Clearly better, not best child but most visits. Stop boosting now. Visits for best child "
		      << best_child->GetN() << " visits in flight for best child: " << best_child->GetNInFlight()
		      << " visits for boosted node: " << boosted_node->GetN() << " visits in flight for boosted node: "
		      << boosted_node->GetNInFlight();
	    }
	    collision_limit_one = 0;
	    // collision_limit_one = std::max(collision_limit * 1 / 3, static_cast<int>(std::floor(collision_limit * params_.GetAuxEngineForceVisitsRatio())));
	    // collision_limit_one = collision_limit * params_.GetAuxEngineForceVisitsRatio();
	    // keep the default, all visits go here.
	  } else {
	    LOGFILE << "Case 1: Clearly better, and not yet best child.";
	  }
	}
	// So, collision_limit_one is now set appropriately by comparing the first node in this sub-tree with its best sibling.
	// Now, change what node to actually boost from the first node in this branch to (1) some relevant node or (2) directly use the leaf in the helper's PV.
	// First, try (1), mostly because for (2) we do not have the node yet (only the vector of moves).
	boosted_node = search_->search_stats_->helper_PV_from_instance_two_explore_node;
	vector_of_moves_from_root_to_boosted_node = search_->search_stats_->helper_PV_from_instance_two_explore_moves;
      }

      if(collision_limit_one > 0){

	search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_.unlock();
	search_->search_stats_->best_move_candidates_mutex.unlock_shared();
	  
	// This is irrelevant when boosting the second divergence
	if(donate_visits){
	  if(vector_of_moves_from_root_to_boosted_node.size() == 0){
	    LOGFILE << "This should never happen. Second divergence is at depth 1, so parent of that node is root, stop boosting here.";
	//   } else {
	//     // LOGFILE << "Second divergence is at depth: " << vector_of_moves_from_root_to_boosted_node.size()	    
	//     LOGFILE << "Second divergence is at depth: " << divergence_at_depth
	// 	    << " Forcing " << collision_limit_one << " visits at node at depth "
	// 	    << vector_of_moves_from_root_to_boosted_node.size() << ", which has "
	// 	    << boosted_node->GetN() << " visits and " << boosted_node->GetNInFlight() << " visits in flight.";
	//   }
	// } else {
	//   if(boosted_node != best_child && boosted_node->GetN() + boosted_node->GetNInFlight() < best_child->GetN() + best_child->GetNInFlight()){
	//     LOGFILE << "Depth: " << vector_of_moves_from_root_to_boosted_node.size() << " Visits for best child (cpuct=1): "
	// 	    << best_child->GetN() << " visits for boosted_node: " << boosted_node->GetN()
	// 	    << " visits in flight for boosted node: " << boosted_node->GetNInFlight()
	// 	    << " collisions left: " << collision_limit;
	//   } else {
	//     LOGFILE << "Depth: " << vector_of_moves_from_root_to_boosted_node.size() << " Already best child, visits for boosted_node: " << boosted_node->GetN()
	// 	    << " visits in flight for boosted node: " << boosted_node->GetNInFlight() << " collisions left: " << collision_limit;
	  }
	}

	// LOGFILE << "Second divergence is at depth: " << vector_of_moves_from_root_to_boosted_node.size()	    
	LOGFILE << "Divergence is at depth: " << divergence_at_depth << " Visits missing before equality: "
		<< visits_missing_to_be_equal << ", visits missing before equality including visits in flight: "
		<< visits_missing_to_be_equal_including_in_flight
		<< " Forcing " << collision_limit_one << " visits at a node at depth "
		<< vector_of_moves_from_root_to_boosted_node.size() << ", which has "
		<< boosted_node->GetN() << " visits and " << boosted_node->GetNInFlight() << " visits in flight.";

	{

	  Mutex::Lock lock(picking_tasks_mutex_);
	  picking_tasks_.emplace_back(
				      boosted_node,
				      vector_of_moves_from_root_to_boosted_node.size(),
				      vector_of_moves_from_root_to_boosted_node,
				      collision_limit_one, probability_of_best_path, distance_from_best_path);
	  task_count_.fetch_add(1, std::memory_order_acq_rel);
	  task_added_.notify_all();
	}
	WaitForTasks();	  

	// Add a VisitInFlight for every non_collision
	for(Node * n = boosted_node; n != search_->root_node_; n = n->GetParent()){
	  n->IncrementNInFlight(collision_limit_one);
	}
	// // The loop above stops just before root, so fix root too. // TODO fix this ugly off-by-one hack. (perhaps test for n != nullptr)
	search_->root_node_->IncrementNInFlight(collision_limit_one);
	return true;
      } // End of collision_limit_one > 0
      else {
	LOGFILE << " collision_limit for boosted node turned out to be zero.";
      }
    } else { // End of "no reason to enforce visits".
      // Don't spam the log when autopilot is on.
      if(override_cpuct == 1 && !search_->search_stats_->winning_){
	std::string why = "";
	if(! (search_->search_stats_->number_of_nodes_in_support_for_helper_eval_of_leelas_preferred_child > 0)) why = why + " too few nodes in support of instance one " ;
	if(! (search_->search_stats_->Helpers_preferred_child_node_in_Leelas_PV_ != nullptr)) why = why + " no second divergence node yet " ;
	if(! (search_->search_stats_->Helpers_preferred_child_node_ != nullptr)) why = why + " no first divergence node yet " ;
	if(search_->search_stats_->Helpers_preferred_child_node_ != nullptr && search_->search_stats_->Helpers_preferred_child_node_->GetN() == 0) why = why + " the first divergence as zero visits ";
	if(! (search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_.size() > 0)) why = why + " zero sized vector to helper's preferred child " ;
	if(! (search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_in_Leelas_PV_.size() > 0)) why = why + " zero sized vector to helper's preferred child in leelas pv " ;
	LOGFILE << " Not forcing visits this batch because: " << why;
      }
    }
    search_->search_stats_->vector_of_moves_from_root_to_Helpers_preferred_child_node_mutex_.unlock();
    search_->search_stats_->best_move_candidates_mutex.unlock_shared();
  } // end of override CPUCT
  
  auto& vtp_buffer = workspace->vtp_buffer;
  auto& visits_to_perform = workspace->visits_to_perform;
  visits_to_perform.clear();
  auto& vtp_last_filled = workspace->vtp_last_filled;
  vtp_last_filled.clear();
  auto& current_path = workspace->current_path;
  current_path.clear();
  auto& moves_to_path = workspace->moves_to_path;
  moves_to_path = moves_to_base;
  // Sometimes receiver is reused, othertimes not, so only jump start if small.
  if (receiver->capacity() < 30) {
    receiver->reserve(receiver->size() + 30);
  }

  // These 2 are 'filled pre-emptively'.
  std::array<float, 256> current_pol;
  std::array<float, 256> current_util;

  // These 3 are 'filled on demand'.
  std::array<float, 256> current_score;
  std::array<int, 256> current_nstarted;
  auto& cur_iters = workspace->cur_iters;

  Node::Iterator best_edge;
  Node::Iterator second_best_edge;
  // Fetch the current best root node visits for possible smart pruning.
  const int64_t best_node_n = search_->current_best_edge_.GetN();

  int passed_off = 0;
  int completed_visits = 0;

  bool is_root_node = node == search_->root_node_;
  const float even_draw_score = search_->GetDrawScore(false);
  const float odd_draw_score = search_->GetDrawScore(true);
  const auto& root_move_filter = search_->root_move_filter_;
  auto m_evaluator = moves_left_support_ ? MEvaluator(params_) : MEvaluator();

  int max_limit = std::numeric_limits<int>::max();

  current_path.push_back(-1);
  while (current_path.size() > 0) {
    // First prepare visits_to_perform.
    if (current_path.back() == -1) {
      // Need to do n visits, where n is either collision_limit, or comes from
      // visits_to_perform for the current path.
      int cur_limit = collision_limit;
      if (current_path.size() > 1) {
        cur_limit =
            (*visits_to_perform.back())[current_path[current_path.size() - 2]];
      }
      // First check if node is terminal or not-expanded.  If either than create
      // a collision of appropriate size and pop current_path.
      if (node->GetN() == 0 || node->IsTerminal()) {
        if (is_root_node) {
          // Root node is special - since its not reached from anywhere else, so
          // it needs its own logic. Still need to create the collision to
          // ensure the outer gather loop gives up.
          if (node->TryStartScoreUpdate()) {
            cur_limit -= 1;
            minibatch_.push_back(NodeToProcess::Visit(
                node, static_cast<uint16_t>(current_path.size() + base_depth)));
            completed_visits++;
          }
        }
        // Visits are created elsewhere, just need the collisions here.
        if (cur_limit > 0) {
          int max_count = 0;
          if (cur_limit == collision_limit && base_depth == 0 &&
              max_limit > cur_limit) {
            max_count = max_limit;
          }
          receiver->push_back(NodeToProcess::Collision(
              node, static_cast<uint16_t>(current_path.size() + base_depth),
              cur_limit, max_count));
          completed_visits += cur_limit;
        }
        node = node->GetParent();
        current_path.pop_back();
        continue;
      }
      if (is_root_node) {
        // Root node is again special - needs its n in flight updated separately
        // as its not handled on the path to it, since there isn't one.
        node->IncrementNInFlight(cur_limit);
      }

      // Create visits_to_perform new back entry for this level.
      if (vtp_buffer.size() > 0) {
        visits_to_perform.push_back(std::move(vtp_buffer.back()));
        vtp_buffer.pop_back();
      } else {
        visits_to_perform.push_back(std::make_unique<std::array<int, 256>>());
      }
      vtp_last_filled.push_back(-1);

      // Cache all constant UCT parameters.
      int max_needed = node->GetNumEdges();
      node->CopyPolicy(max_needed, current_pol.data());
      for (int i = 0; i < max_needed; i++) {
        current_util[i] = std::numeric_limits<float>::lowest();
      }
      // Root depth is 1 here, while for GetDrawScore() it's 0-based, that's why
      // the weirdness.
      const float draw_score = ((current_path.size() + base_depth) % 2 == 0)
                                   ? odd_draw_score
                                   : even_draw_score;
      m_evaluator.SetParent(node);
      float visited_pol = 0.0f;
      float best_q = -1.0f;
      int number_of_visited_nodes = 0;
      for (Node* child : node->VisitedNodes()) {
	number_of_visited_nodes++;
        int index = child->Index();
        visited_pol += current_pol[index];
        float q = child->GetQ(draw_score);
	if(q > best_q){
	  best_q = q;
	}
        current_util[index] = q + m_evaluator.GetM(child, q);
      }
      // if(number_of_visited_nodes > 1){
      // 	LOGFILE << "Node " << node->DebugString() << " has at least two visited nodes: calculate probabiliy of best path";
      // }
      
      const float fpu =
          GetFpu(params_, node, is_root_node, draw_score, visited_pol);
      for (int i = 0; i < max_needed; i++) {
        if (current_util[i] == std::numeric_limits<float>::lowest()) {
          current_util[i] = fpu + m_evaluator.GetDefaultM();
        }
      }

      const float cpuct = ComputeCpuct(params_, node->GetN(), is_root_node);
      const float puct_mult =
          cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
      int cache_filled_idx = -1;
      while (cur_limit > 0) {
        // Perform UCT for current node.
        float best = std::numeric_limits<float>::lowest();
        int best_idx = -1;
        float best_without_u = std::numeric_limits<float>::lowest();
        float second_best = std::numeric_limits<float>::lowest();
        best_edge.Reset();
        for (int idx = 0; idx < max_needed; ++idx) {
          if (idx > cache_filled_idx) {
            if (idx == 0) {
              cur_iters[idx] = node->Edges();
            } else {
              cur_iters[idx] = cur_iters[idx - 1];
              ++cur_iters[idx];
            }
            current_nstarted[idx] = cur_iters[idx].GetNStarted();
          }
          int nstarted = current_nstarted[idx];
          const float util = current_util[idx];
          if (idx > cache_filled_idx) {
            current_score[idx] =
                current_pol[idx] * puct_mult / (1 + nstarted) + util;
            cache_filled_idx++;
          }
          if (is_root_node) {
	    // If params_.GetQBasedMoveSelection() is false and 
            // there's no chance to catch up to the current best node with
            // remaining playouts, don't consider it.
            // best_move_node_ could have changed since best_node_n was
            // retrieved. To ensure we have at least one node to expand, always
            // include current best node.
            if (!params_.GetQBasedMoveSelection() && cur_iters[idx] != search_->current_best_edge_ &&
                latest_time_manager_hints_.GetEstimatedRemainingPlayouts() <
                    best_node_n - cur_iters[idx].GetN()) {
              continue;
            }
            // If root move filter exists, make sure move is in the list.
            if (!root_move_filter.empty() &&
                std::find(root_move_filter.begin(), root_move_filter.end(),
                          cur_iters[idx].GetMove()) == root_move_filter.end()) {
              continue;
            }
          }

          float score = current_score[idx];
          if (score > best) {
            second_best = best;
            second_best_edge = best_edge;
            best = score;
            best_idx = idx;
            best_without_u = util;
            best_edge = cur_iters[idx];
          } else if (score > second_best) {
            second_best = score;
            second_best_edge = cur_iters[idx];
          }
        }

	// Hack the scores if the child with highest expected Q does not have most visits, ie boost exploration of that child.
	// TODO let this work in-tree with a vector of edge indices.

	if(is_root_node && params_.GetQBasedMoveSelection() &&
	   this_edge_has_higher_expected_q_than_the_most_visited_child > -1){
	  if(this_edge_has_higher_expected_q_than_the_most_visited_child != best_idx){
	    best_idx = this_edge_has_higher_expected_q_than_the_most_visited_child;
	    best_edge = cur_iters[best_idx];
	  }
	}

        int new_visits = 0;
	// easiest is to give the promising node all visits
        if (second_best_edge && (this_edge_has_higher_expected_q_than_the_most_visited_child == -1)) {
          int estimated_visits_to_change_best = std::numeric_limits<int>::max();
          if (best_without_u < second_best) {
            const auto n1 = current_nstarted[best_idx] + 1;
            estimated_visits_to_change_best = static_cast<int>(
                std::max(1.0f, std::min(current_pol[best_idx] * puct_mult /
                                                (second_best - best_without_u) -
                                            n1 + 1,
                                        1e9f)));
          }
          second_best_edge.Reset();
          max_limit = std::min(max_limit, estimated_visits_to_change_best);
          new_visits = std::min(cur_limit, estimated_visits_to_change_best);
        } else {
          // No second best - only one edge, so everything goes in here.
          new_visits = cur_limit;
        }
        if (best_idx >= vtp_last_filled.back()) {
          auto* vtp_array = visits_to_perform.back().get()->data();
          std::fill(vtp_array + (vtp_last_filled.back() + 1),
                    vtp_array + best_idx + 1, 0);
        }
        (*visits_to_perform.back())[best_idx] += new_visits;
        cur_limit -= new_visits;
        Node* child_node = best_edge.GetOrSpawnNode(/* parent */ node);

        // Probably best place to check for two-fold draws consistently.
        // Depth starts with 1 at root, so real depth is depth - 1.
        EnsureNodeTwoFoldCorrectForDepth(
            child_node, current_path.size() + base_depth + 1 - 1);

        bool decremented = false;
        if (child_node->TryStartScoreUpdate()) {
          current_nstarted[best_idx]++;
          new_visits -= 1;
          decremented = true;
          if (child_node->GetN() > 0 && !child_node->IsTerminal()) {
            child_node->IncrementNInFlight(new_visits);
            current_nstarted[best_idx] += new_visits;
          }
          current_score[best_idx] = current_pol[best_idx] * puct_mult /
                                        (1 + current_nstarted[best_idx]) +
                                    current_util[best_idx];
        }
        if ((decremented &&
             (child_node->GetN() == 0 || child_node->IsTerminal()))) {
          // Reduce 1 for the visits_to_perform to ensure the collision created
          // doesn't include this visit.
          (*visits_to_perform.back())[best_idx] -= 1;
          receiver->push_back(NodeToProcess::Visit(
              child_node,
              static_cast<uint16_t>(current_path.size() + 1 + base_depth)));
          completed_visits++;
	  
	  // LOGFILE << "At a leaf: Parent " << node->DebugString() << " Child " << node->DebugString() << " setting probability of best path to: " << probability_of_best_path;
	  receiver->back().best_path_probability = probability_of_best_path;
          receiver->back().moves_to_visit.reserve(moves_to_path.size() + 1);
          receiver->back().moves_to_visit = moves_to_path;
          receiver->back().moves_to_visit.push_back(best_edge.GetMove());
        }
        if (best_idx > vtp_last_filled.back() &&
            (*visits_to_perform.back())[best_idx] > 0) {
          vtp_last_filled.back() = best_idx;
        }
      }
      is_root_node = false;
      // Actively do any splits now rather than waiting for potentially long
      // tree walk to get there.
      for (int i = 0; i <= vtp_last_filled.back(); i++) {
        int child_limit = (*visits_to_perform.back())[i];
        if (params_.GetTaskWorkersPerSearchWorker() > 0 &&
            child_limit > params_.GetMinimumWorkSizeForPicking() &&
            child_limit <
                ((collision_limit - passed_off - completed_visits) * 2 / 3) &&
            child_limit + passed_off + completed_visits <
                collision_limit -
                    params_.GetMinimumRemainingWorkSizeForPicking()) {
          Node* child_node = cur_iters[i].GetOrSpawnNode(/* parent */ node);
          // Don't split if not expanded or terminal.
          if (child_node->GetN() == 0 || child_node->IsTerminal()) continue;

          bool passed = false;
          {
            // Multiple writers, so need mutex here.
            Mutex::Lock lock(picking_tasks_mutex_);
            // Ensure not to exceed size of reservation.
            if (picking_tasks_.size() < MAX_TASKS) {
	      float new_probability_of_best_path = (1-(best_q - child_node->GetQ(draw_score))) * probability_of_best_path;
	      // LOGFILE << "Calculating probability of best path... q=" << child_node->GetQ(draw_score)
	      // 	      << " best_q: " << best_q << " diff: " << best_q - child_node->GetQ(draw_score)
	      // 	      << " 1-diff: " << 1-(best_q - child_node->GetQ(draw_score))
	      // 	      << " old probability: " << probability_of_best_path
	      // 	      << " 1-diff times the old probability: " << new_probability_of_best_path;
              moves_to_path.push_back(cur_iters[i].GetMove());
	      if(new_probability_of_best_path != probability_of_best_path){
		distance_from_best_path++;
	      }
              picking_tasks_.emplace_back(
                  child_node, current_path.size() - 1 + base_depth + 1,
                  moves_to_path, child_limit, new_probability_of_best_path, distance_from_best_path);
              moves_to_path.pop_back();
              task_count_.fetch_add(1, std::memory_order_acq_rel);
              task_added_.notify_all();
              passed = true;
              passed_off += child_limit;
            }
          }
          if (passed) {
            (*visits_to_perform.back())[i] = 0;
          }
        }
      }
      // Fall through to select the first child.
    }
    int min_idx = current_path.back();
    bool found_child = false;
    if (vtp_last_filled.back() > min_idx) {
      int idx = -1;
      for (auto& child : node->Edges()) {
        idx++;
        if (idx > min_idx && (*visits_to_perform.back())[idx] > 0) {
          if (moves_to_path.size() != current_path.size() + base_depth) {
            moves_to_path.push_back(child.GetMove());
          } else {
            moves_to_path.back() = child.GetMove();
          }
          current_path.back() = idx;
          current_path.push_back(-1);
          node = child.GetOrSpawnNode(/* parent */ node);
          found_child = true;
          break;
        }
        if (idx >= vtp_last_filled.back()) break;
      }
    }
    if (!found_child) {
      node = node->GetParent();
      if (!moves_to_path.empty()) moves_to_path.pop_back();
      current_path.pop_back();
      vtp_buffer.push_back(std::move(visits_to_perform.back()));
      visits_to_perform.pop_back();
      vtp_last_filled.pop_back();
    }
  }
  return true;
}

void SearchWorker::ExtendNode(Node* node, int depth,
                              const std::vector<Move>& moves_to_node,
                              PositionHistory* history) {

  // Initialize position sequence with pre-move position.
  history->Trim(search_->played_history_.GetLength());
  for (size_t i = 0; i < moves_to_node.size(); i++) {
    history->Append(moves_to_node[i]);
  }

  // We don't need the mutex because other threads will see that N=0 and
  // N-in-flight=1 and will not touch this node.
  const auto& board = history->Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();

  // Check whether it's a draw/lose by position. Importantly, we must check
  // these before doing the by-rule checks below.
  if (legal_moves.empty()) {
    // Could be a checkmate or a stalemate
    if (board.IsUnderCheck()) {
      node->MakeTerminal(GameResult::WHITE_WON);
    } else {
      node->MakeTerminal(GameResult::DRAW);
    }
    return;
  }

  // We can shortcircuit these draws-by-rule only if they aren't root;
  // if they are root, then thinking about them is the point.
  if (node != search_->root_node_) {
    if (!board.HasMatingMaterial()) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history->Last().GetRule50Ply() >= 100) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    const auto repetitions = history->Last().GetRepetitions();
    // Mark two-fold repetitions as draws according to settings.
    // Depth starts with 1 at root, so number of plies in PV is depth - 1.
    if (repetitions >= 2) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    } else if (repetitions == 1 && depth - 1 >= 4 &&
               params_.GetTwoFoldDraws() &&
               depth - 1 >= history->Last().GetPliesSincePrevRepetition()) {
      const auto cycle_length = history->Last().GetPliesSincePrevRepetition();
      // use plies since first repetition as moves left; exact if forced draw.
      node->MakeTerminal(GameResult::DRAW, (float)cycle_length,
                         Node::Terminal::TwoFold);
      return;
    }

    // Neither by-position or by-rule termination, but maybe it's a TB position.
    if (search_->syzygy_tb_ && !search_->root_is_in_dtz_ &&
        board.castlings().no_legal_castle() &&
        history->Last().GetRule50Ply() == 0 &&
        (board.ours() | board.theirs()).count() <=
            search_->syzygy_tb_->max_cardinality()) {
      ProbeState state;
      const WDLScore wdl =
          search_->syzygy_tb_->probe_wdl(history->Last(), &state);
      // Only fail state means the WDL is wrong, probe_wdl may produce correct
      // result with a stat other than OK.
      if (state != FAIL) {
        // TB nodes don't have NN evaluation, assign M from parent node.
        float m = 0.0f;
        // Need a lock to access parent, in case MakeSolid is in progress.
        {
          SharedMutex::SharedLock lock(search_->nodes_mutex_);
          auto parent = node->GetParent();
          if (parent) {
            m = std::max(0.0f, parent->GetM() - 1.0f);
          }
        }
        // If the colors seem backwards, check the checkmate check above.
        if (wdl == WDL_WIN) {
          node->MakeTerminal(GameResult::BLACK_WON, m,
                             Node::Terminal::Tablebase);
        } else if (wdl == WDL_LOSS) {
          node->MakeTerminal(GameResult::WHITE_WON, m,
                             Node::Terminal::Tablebase);
        } else {  // Cursed wins and blessed losses count as draws.
          node->MakeTerminal(GameResult::DRAW, m, Node::Terminal::Tablebase);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }
  }

  // Add legal moves as edges of this node.
  node->CreateEdges(legal_moves);
}

// Returns whether node was already in cache.
bool SearchWorker::AddNodeToComputation(Node* node) {
  const auto hash = history_.HashLast(params_.GetCacheHistoryLength() + 1);
  if (search_->cache_->ContainsKey(hash)) {
    return true;
  }
  int transform;
  auto planes =
      EncodePositionForNN(search_->network_->GetCapabilities().input_format,
                          history_, 8, params_.GetHistoryFill(), &transform);

  std::vector<uint16_t> moves;

  if (node && node->HasChildren()) {
    // Legal moves are known, use them.
    moves.reserve(node->GetNumEdges());
    for (const auto& edge : node->Edges()) {
      moves.emplace_back(edge.GetMove().as_nn_index(transform));
    }
  } else {
    // Cache pseudolegal moves. A bit of a waste, but faster.
    const auto& pseudolegal_moves =
        history_.Last().GetBoard().GeneratePseudolegalMoves();
    moves.reserve(pseudolegal_moves.size());
    for (auto iter = pseudolegal_moves.begin(), end = pseudolegal_moves.end();
         iter != end; ++iter) {
      moves.emplace_back(iter->as_nn_index(transform));
    }
  }

  computation_->AddInput(hash, std::move(planes), std::move(moves));
  return false;
}

// 2b. Copy collisions into shared collisions.
void SearchWorker::CollectCollisions() {
  SharedMutex::Lock lock(search_->nodes_mutex_);

  for (const NodeToProcess& node_to_process : minibatch_) {
    if (node_to_process.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_to_process.node,
                                               node_to_process.multivisit);
    }
  }
}

// 3. Prefetch into cache.
// ~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::MaybePrefetchIntoCache() {
  // TODO(mooskagh) Remove prefetch into cache if node collisions work well.
  // If there are requests to NN, but the batch is not full, try to prefetch
  // nodes which are likely useful in future.
  if (search_->stop_.load(std::memory_order_acquire)) return;
  if (computation_->GetCacheMisses() > 0 &&
      computation_->GetCacheMisses() < params_.GetMaxPrefetchBatch()) {
    history_.Trim(search_->played_history_.GetLength());
    SharedMutex::SharedLock lock(search_->nodes_mutex_);
    PrefetchIntoCache(
        search_->root_node_,
        params_.GetMaxPrefetchBatch() - computation_->GetCacheMisses(), false);
  }
}

// Prefetches up to @budget nodes into cache. Returns number of nodes
// prefetched.
int SearchWorker::PrefetchIntoCache(Node* node, int budget, bool is_odd_depth) {
  const float draw_score = search_->GetDrawScore(is_odd_depth);
  if (budget <= 0) return 0;

  // We are in a leaf, which is not yet being processed.
  if (!node || node->GetNStarted() == 0) {
    if (AddNodeToComputation(node)) {
      // Make it return 0 to make it not use the slot, so that the function
      // tries hard to find something to cache even among unpopular moves.
      // In practice that slows things down a lot though, as it's not always
      // easy to find what to cache.
      return 1;
    }
    return 1;
  }

  assert(node);
  // n = 0 and n_in_flight_ > 0, that means the node is being extended.
  if (node->GetN() == 0) return 0;
  // The node is terminal; don't prefetch it.
  if (node->IsTerminal()) return 0;

  // Populate all subnodes and their scores.
  typedef std::pair<float, EdgeAndNode> ScoredEdge;
  std::vector<ScoredEdge> scores;
  const float cpuct =
      ComputeCpuct(params_, node->GetN(), node == search_->root_node_);
  const float puct_mult =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  const float fpu =
      GetFpu(params_, node, node == search_->root_node_, draw_score);
  for (auto& edge : node->Edges()) {
    if (edge.GetP() == 0.0f) continue;
    // Flip the sign of a score to be able to easily sort.
    // TODO: should this use logit_q if set??
    scores.emplace_back(-edge.GetU(puct_mult) - edge.GetQ(fpu, draw_score),
                        edge);
  }

  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;  // Initialize for the case where there's only
                                 // one child.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (search_->stop_.load(std::memory_order_acquire)) break;
    if (budget <= 0) break;

    // Sort next chunk of a vector. 3 at a time. Most of the time it's fine.
    if (first_unsorted_index != scores.size() &&
        i + 2 >= first_unsorted_index) {
      const int new_unsorted_index =
          std::min(scores.size(), budget < 2 ? first_unsorted_index + 2
                                             : first_unsorted_index + 3);
      std::partial_sort(scores.begin() + first_unsorted_index,
                        scores.begin() + new_unsorted_index, scores.end(),
                        [](const ScoredEdge& a, const ScoredEdge& b) {
                          return a.first < b.first;
                        });
      first_unsorted_index = new_unsorted_index;
    }

    auto edge = scores[i].second;
    // Last node gets the same budget as prev-to-last node.
    if (i != scores.size() - 1) {
      // Sign of the score was flipped for sorting, so flip it back.
      const float next_score = -scores[i + 1].first;
      // TODO: As above - should this use logit_q if set?
      const float q = edge.GetQ(-fpu, draw_score);
      if (next_score > q) {
        budget_to_spend =
            std::min(budget, int(edge.GetP() * puct_mult / (next_score - q) -
                                 edge.GetNStarted()) +
                                 1);
      } else {
        budget_to_spend = budget;
      }
    }
    history_.Append(edge.GetMove());
    const int budget_spent =
        PrefetchIntoCache(edge.node(), budget_to_spend, !is_odd_depth);
    history_.Pop();
    budget -= budget_spent;
    total_budget_spent += budget_spent;
  }
  return total_budget_spent;
}

// 4. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() { computation_->ComputeBlocking(); }

// 5. Retrieve NN computations (and terminal values) into nodes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::FetchMinibatchResults() {
  // Populate NN/cached results, or terminal results, into nodes.
  int idx_in_computation = 0;
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process, *computation_, idx_in_computation);
    if (node_to_process.nn_queried) ++idx_in_computation;
  }
}

template <typename Computation>
void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process,
                                         const Computation& computation,
                                         int idx_in_computation) {

  if (node_to_process->IsCollision()) return;
  
  Node* node = node_to_process->node;
  if (!node_to_process->nn_queried) {
    // Terminal nodes don't involve the neural NetworkComputation, nor do
    // they require any further processing after value retrieval.
    node_to_process->v = node->GetWL();
    node_to_process->d = node->GetD();
    node_to_process->m = node->GetM();
    return;
  }
  // For NN results, we need to populate policy as well as value.
  // First the value...
  node_to_process->v = -computation.GetQVal(idx_in_computation);
  node_to_process->d = computation.GetDVal(idx_in_computation);
  node_to_process->m = computation.GetMVal(idx_in_computation);
  // LOGFILE << "v for node: " << node_to_process->node->DebugString() << " is " << node_to_process->v;
  // ...and secondly, the policy data.
  // Calculate maximum first.
  float max_p = -std::numeric_limits<float>::infinity();
  // Intermediate array to store values when processing policy.
  // There are never more than 256 valid legal moves in any legal position.
  std::array<float, 256> intermediate;
  int counter = 0;

  for (auto& edge : node->Edges()) {
    float p = computation.GetPVal(
        idx_in_computation,
        edge.GetMove().as_nn_index(node_to_process->probability_transform));
    intermediate[counter++] = p;
    max_p = std::max(max_p, p);
  }
  float total = 0.0;
  for (int i = 0; i < counter; i++) {
    // Perform softmax and take into account policy softmax temperature T.
    // Note that we want to calculate (exp(p-max_p))^(1/T) = exp((p-max_p)/T).
    float p =
        FastExp((intermediate[i] - max_p) / params_.GetPolicySoftmaxTemp());
    intermediate[i] = p;
    total += p;
  }

  counter = 0;
  // Normalize P values to add up to 1.0.
  const float scale = total > 0.0f ? 1.0f / total : 1.0f;
  for (auto& edge : node->Edges()) {
    edge.edge()->SetP(intermediate[counter++] * scale);
  }
  // Add Dirichlet noise if enabled and at root.
  if (params_.GetNoiseEpsilon() && node == search_->root_node_) {
    ApplyDirichletNoise(node, params_.GetNoiseEpsilon(),
                        params_.GetNoiseAlpha());
  }
}

// 6. Propagate the new nodes' information to all their parents in the tree.
// ~~~~~~~~~~~~~~
void SearchWorker::DoBackupUpdate() {
  // Nodes mutex for doing node updates.
  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << "DoBackupUpdate() trying to get a shared lock on nodes";  
  SharedMutex::Lock lock(search_->nodes_mutex_);
  if (params_.GetAuxEngineVerbosity() >= 10) LOGFILE << "DoBackupUpdate() Aquired shared lock on nodes";

  bool work_done = number_out_of_order_ > 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
    if (!node_to_process.IsCollision()) {
      work_done = true;
    }
  }
  if (!work_done) return;
  search_->CancelSharedCollisions();
  search_->total_batches_ += 1;
}

void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  Node* node = node_to_process.node;
  if (node_to_process.IsCollision()) {
    // Collisions are handled via shared_collisions instead.
    return;
  }

  // For the first visit to a terminal, maybe update parent bounds too.
  auto update_parent_bounds =
      params_.GetStickyEndgames() && node->IsTerminal() && !node->GetN();

  // Backup V value up to a root. After 1 visit, V = Q.
  float v = node_to_process.v;
  float d = node_to_process.d;
  float m = node_to_process.m;
  int n_to_fix = 0;
  float v_delta = 0.0f;
  float d_delta = 0.0f;
  float m_delta = 0.0f;
  uint32_t solid_threshold =
      static_cast<uint32_t>(params_.GetSolidTreeThreshold());

  float probability_of_best_path = node_to_process.best_path_probability;
  // LOGFILE << "BackupUpdate: probability_of_best_path: " << probability_of_best_path;
  // LOGFILE << "BackupUpdate: depth of node to process: " << node_to_process.depth;

  // no longer used
  std::vector<Move> my_moves;
  int depth = 0;

  // LOGFILE << "Updating node: " << node->DebugString() << " root_node_ is " << search_->root_node_->DebugString() << " and root has MiniMaxQ of " << search_->root_node_->GetBestMiniMaxQ();
  
  for (Node *n = node, *p; n != search_->root_node_->GetParent(); n = p) {
    p = n->GetParent();

    // In order to be able to construct the board store the moves from
    // root to this node. Don't store the move leading to root.
    if(n != search_->root_node_ && depth > 0) { // skip the last move, we want to extend siblings to the last node.
      my_moves.push_back(n->GetOwnEdge()->GetMove());
      depth++;
    } else {
      depth++;      
    }

    // Current node might have become terminal from some other descendant, so
    // backup the rest of the way with more accurate values.
    if (n->IsTerminal()) {
      v = n->GetWL();
      d = n->GetD();
      m = n->GetM();
    }
    n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit, node_to_process.depth - depth);
    if (n_to_fix > 0 && !n->IsTerminal()) {
      n->AdjustForTerminal(v_delta, d_delta, m_delta, n_to_fix);
    }
    if (n->GetN() >= solid_threshold) {
      if (n->MakeSolid() && n == search_->root_node_) {
        // If we make the root solid, the current_best_edge_ becomes invalid and
        // we should repopulate it.
        search_->current_best_edge_ =
            search_->GetBestChildNoTemperature(search_->root_node_, 0);
      }
    }

    // Nothing left to do without ancestors to update.
    if (!p) break;

    bool old_update_parent_bounds = update_parent_bounds;
    // If parent already is terminal further adjustment is not required.
    if (p->IsTerminal()) n_to_fix = 0;
    // Try setting parent bounds except the root or those already terminal.
    update_parent_bounds =
        update_parent_bounds && p != search_->root_node_ && !p->IsTerminal() &&
        MaybeSetBounds(p, m, &n_to_fix, &v_delta, &d_delta, &m_delta);

    // Q will be flipped for opponent.
    v = -v;
    v_delta = -v_delta;
    m++;

    // Update the stats.
    // Best move.
    // If update_parent_bounds was set, we just adjusted bounds on the
    // previous loop or there was no previous loop, so if n is a terminal, it
    // just became that way and could be a candidate for changing the current
    // best edge. Otherwise a visit can only change best edge if its to an edge
    // that isn't already the best and the new n is equal or greater to the old
    // n.
    if (p == search_->root_node_ &&
        ((old_update_parent_bounds && n->IsTerminal()) ||
         (n != search_->current_best_edge_.node() &&
          search_->current_best_edge_.GetN() <= n->GetN()))) {
      search_->current_best_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_, 0);
    }

    // Avoid a full function call unless it will likely actually queue the node.
    // Do nothing if search is interrupted, the node will get picked the next iteration anyway.

    // Not taking a lock here since it would be expensive AuxEngineThreshold is never adjusted during search anyway.

    if(search_->search_stats_->AuxEngineThreshold > 0 &&
       n->GetN() >= (uint32_t) search_->search_stats_->AuxEngineThreshold &&
       n->GetAuxEngineMove() == 0xffff &&
      !n->IsTerminal() &&
       n->HasChildren() &&
       // These last two conditions are rather expensive to evaluate, which is why they must come last
       params_.GetAuxEngineFile() != ""
       ){
      // AuxMaybeEnqueueNode(n, 1);
      AuxMaybeEnqueueNode(n);      
    }
  }

  // Quiescence search:

  // Case A: Check if Q-delta between parent and best policy child is lower than some threshold, otherwise put the highest policy node in the preextend-queue right away.
  // If parent node has two visits, then this node must be its child with highest policy.
  if(node != search_->root_node_ && node->GetParent()->GetN() == 2 && !node->IsTerminal() && params_.GetAuxEngineFile() != ""){
    float q_of_parent = node->GetParent()->GetQ(0.0f);
    float q_of_node = node->GetQ(0.0f);
    // float delta = std::abs(q_of_node - q_of_parent); // since they have opposite signs, adding works fine here.
    float delta = std::abs(q_of_node + q_of_parent); // since they have opposite signs, adding works fine here.    
    if(delta * probability_of_best_path > params_.GetQuiescenceDeltaThreshold() &&
       q_of_node < q_of_parent &&
       delta * pow(probability_of_best_path, params_.GetQuiescencePolicyThreshold()) > params_.GetQuiescenceDeltaThreshold() ){
      // Create a vector with elements of type Move from root to this node and queue that vector.
      std::vector<lczero::Move> my_moves_from_the_white_side;
      // Add best child
      float highest_p = 0;
      Edge * this_edge_has_highest_p;	  
      // loop through the policies of the children.
      for (auto& edge : node->Edges()) {
	if(edge.GetP() > highest_p) {
	  highest_p = edge.GetP();
	  this_edge_has_highest_p = edge.edge();
	}
      }
      my_moves_from_the_white_side.push_back(this_edge_has_highest_p->GetMove());
      // Add the rest of the moves.
      for (Node* n2 = node; n2 != search_->root_node_; n2 = n2->GetParent()) {
	my_moves_from_the_white_side.push_back(n2->GetOwnEdge()->GetMove());
      }
      // Reverse the order
      std::reverse(my_moves_from_the_white_side.begin(), my_moves_from_the_white_side.end());

      // Queue the vector
      search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock(); // lock this queue before starting to modify it
      search_->search_stats_->fast_track_extend_and_evaluate_queue_.push(my_moves_from_the_white_side);
      search_->search_stats_->starting_depth_of_PVs_.push(my_moves_from_the_white_side.size());
      search_->search_stats_->amount_of_support_for_PVs_.push(0);
      // Also do some stats
      search_->search_stats_->Number_of_nodes_fast_tracked_because_of_fluctuating_eval++;
      search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();
    }
  }

  // if(node != search_->root_node_ && node->GetParent()->GetN() == 2 && probability_of_best_path > params_.GetQuiescencePolicyThreshold()){

  //   // Case B: Current move was a capture, or check
  //   // my_moves has the moves from root to the non-extended child.
  //   // 1. create the board of the previous position.
  //   // 1b. count pieces
  //   // 2. create the board of the node.
  //   // 2b. count pieces
  //   // If the number of pieces in 2b is not the same as the number of pieces in 1b then there was a capture
  //   // For check, only check the position of the child.
  //   // if(node != search_->root_node_ && !node->IsTerminal() && node->GetOwnEdge()->GetP() > params_.GetQuiescencePolicyThreshold()){
    
  //   // if(delta > params_.GetQuiescencePolicyThreshold() && q_of_node < q_of_parent){    
  //   // if(probability_of_best_path > params_.GetQuiescencePolicyThreshold()){

  //   // reverse the order of the moves
  //     std::reverse(my_moves.begin(), my_moves.end());
  //     Node* parent = node->GetParent();
      
  //     // since this a highely relevant node, automatically extend all its edges.
  //     search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock(); // lock this queue before starting to modify it
  //     for (auto& edge : parent->Edges()) {
  // 	if(!edge.HasNode()){
  // 	  std::vector<lczero::Move> my_moves_copy = my_moves;

  // 	  // check if the PV is new
  // 	  std::ostringstream oss;
  // 	  // Convert all but the last element to avoid a trailing "," https://stackoverflow.com/questions/8581832/converting-a-vectorint-to-string
  // 	  std::copy(pv_moves.begin(), pv_moves.end()-1, std::ostream_iterator<int>(oss, ","));
  // 	  // Now add the last element with no delimiter
  // 	  oss << pv_moves.back();
  // 	  // TODO protect the PV cache with a mutex? Stockfish does not, and worst case scenario is that the same PV is sent again, so probably not needed.
  // 	  // https://stackoverflow.com/questions/8581832/converting-a-vectorint-to-string
  // 	  search_stats_->my_pv_cache_mutex_.lock();
  // 	  if ( search_stats_->my_pv_cache_.find(oss.str()) == search_stats_->my_pv_cache_.end() ) {
  // 	    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "string not found in the cache, adding it.";
  // 	    search_stats_->my_pv_cache_[oss.str()] = true;
  // 	  } else {
  // 	    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "string found in the cache. Return early.";
  // 	    search_stats_->my_pv_cache_mutex_.unlock();
  // 	    return;
  // 	  }
  // 	  search_stats_->my_pv_cache_mutex_.unlock();

	  
  // 	  my_moves_copy.push_back(edge.GetMove());
  // 	  search_->search_stats_->fast_track_extend_and_evaluate_queue_.push(my_moves_copy);
  // 	  search_->search_stats_->starting_depth_of_PVs_.push(my_moves_copy.size());
  // 	  search_->search_stats_->amount_of_support_for_PVs_.push(0);
  // 	}
  //     }
  //     search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();

      // // Require only a small increase in Q to extend the best policy move if it is capture or check
      
      // // Todo, store the board in the minibatch_ so we don't have to traverse all the way from root for all new nodes.
      // // Even better, do this already at picknodestoextend(), there multiple nodes can efficiently share the same tree.

      // ChessBoard my_board = search_->played_history_.Last().GetBoard();
      // if(search_->played_history_.IsBlackToMove()){
      // 	my_board.Mirror();
      // }

      // // reverse the order of the moves
      // std::reverse(my_moves.begin(), my_moves.end());
      // int number_of_pieces_before;
      // int number_of_pieces_now;
      // long unsigned int counter=0;
      // // apply the moves to construct the board
      // for(auto& move: my_moves) {
      // 	counter++;
      // 	my_board.ApplyMove(move);
      // 	my_board.Mirror();
      // 	if(counter == my_moves.size()-1){
      // 	  number_of_pieces_before = my_board.ours().count() + my_board.theirs().count();
      // 	}
      // 	if(counter == my_moves.size()){
      // 	  number_of_pieces_now = my_board.ours().count() + my_board.theirs().count();
      // 	}
      // }
      // if(number_of_pieces_before > number_of_pieces_now || my_board.IsUnderCheck()){
      
      // 	// find best child
      // 	float highest_p = 0;
      // 	Edge * this_edge_has_highest_p;	  
      // 	// loop through the policies of the children.
      // 	for (auto& edge : node->Edges()) {
      // 	  if(edge.GetP() > highest_p) {
      // 	    highest_p = edge.GetP();
      // 	    this_edge_has_highest_p = edge.edge();
      // 	  }
      // 	}

      // 	// Now check if this edge also involves a capture or check
      // 	my_board.ApplyMove(this_edge_has_highest_p->GetMove());
      // 	int number_of_pieces_in_the_future = my_board.ours().count() + my_board.theirs().count();
      // 	if(number_of_pieces_in_the_future < number_of_pieces_now || my_board.IsUnderCheck()){
      // 	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "adding node due to check or capture" << " policy: " << this_edge_has_highest_p->GetP() << " future: " << number_of_pieces_in_the_future << " past: " << number_of_pieces_before;
      // 	  // Add this move to the queue.
      // 	  std::vector<lczero::Move> my_moves_copy = my_moves;
      // 	  my_moves_copy.push_back(this_edge_has_highest_p->GetMove());
      // 	  // Queue the vector
      // 	  search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.lock(); // lock this queue before starting to modify it
      // 	  search_->search_stats_->fast_track_extend_and_evaluate_queue_.push(my_moves_copy);
      // 	  search_->search_stats_->starting_depth_of_PVs_.push(my_moves_copy.size());
      // 	  search_->search_stats_->amount_of_support_for_PVs_.push(0);
      // 	  search_->search_stats_->fast_track_extend_and_evaluate_queue_mutex_.unlock();
      // 	}
      // }
  
  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ += node_to_process.depth * node_to_process.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);

}

bool SearchWorker::MaybeSetBounds(Node* p, float m, int* n_to_fix,
                                  float* v_delta, float* d_delta,
                                  float* m_delta) const {
  auto losing_m = 0.0f;
  auto prefer_tb = false;

  // Determine the maximum (lower, upper) bounds across all children.
  // (-1,-1) Loss (initial and lowest bounds)
  // (-1, 0) Can't Win
  // (-1, 1) Regular node
  // ( 0, 0) Draw
  // ( 0, 1) Can't Lose
  // ( 1, 1) Win (highest bounds)
  auto lower = GameResult::BLACK_WON;
  auto upper = GameResult::BLACK_WON;
  for (const auto& edge : p->Edges()) {
    const auto [edge_lower, edge_upper] = edge.GetBounds();
    lower = std::max(edge_lower, lower);
    upper = std::max(edge_upper, upper);

    // Checkmate is the best, so short-circuit.
    const auto is_tb = edge.IsTbTerminal();
    if (edge_lower == GameResult::WHITE_WON && !is_tb) {
      prefer_tb = false;
      break;
    } else if (edge_upper == GameResult::BLACK_WON) {
      // Track the longest loss.
      losing_m = std::max(losing_m, edge.GetM(0.0f));
    }
    prefer_tb = prefer_tb || is_tb;
  }

  // The parent's bounds are flipped from the children (-max(U), -max(L))
  // aggregated as if it was a single child (forced move) of the same bound.
  //       Loss (-1,-1) -> ( 1, 1) Win
  //  Can't Win (-1, 0) -> ( 0, 1) Can't Lose
  //    Regular (-1, 1) -> (-1, 1) Regular
  //       Draw ( 0, 0) -> ( 0, 0) Draw
  // Can't Lose ( 0, 1) -> (-1, 0) Can't Win
  //        Win ( 1, 1) -> (-1,-1) Loss

  // Nothing left to do for ancestors if the parent would be a regular node.
  if (lower == GameResult::BLACK_WON && upper == GameResult::WHITE_WON) {
    return false;
  } else if (lower == upper) {
    // Search can stop at the parent if the bounds can't change anymore, so make
    // it terminal preferring shorter wins and longer losses.
    *n_to_fix = p->GetN();
    assert(*n_to_fix > 0);
    float cur_v = p->GetWL();
    float cur_d = p->GetD();
    float cur_m = p->GetM();
    p->MakeTerminal(
        -upper,
        (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m) + 1.0f,
        prefer_tb ? Node::Terminal::Tablebase : Node::Terminal::EndOfGame);
    // Negate v_delta because we're calculating for the parent, but immediately
    // afterwards we'll negate v_delta in case it has come from the child.
    *v_delta = -(p->GetWL() - cur_v);
    *d_delta = p->GetD() - cur_d;
    *m_delta = p->GetM() - cur_m;
  } else {
    p->SetBounds(-upper, -lower);
  }

  // Bounds were set, so indicate we should check the parent too.
  return true;
}

  // 6.5
void SearchWorker::MaybeAdjustPolicyForHelperAddedNodes(const std::shared_ptr<Search::adjust_policy_stats> foo){
  std::thread::id this_id = std::this_thread::get_id();
  long unsigned int my_queue_size = foo->queue_of_vector_of_nodes_from_helper_added_by_this_thread.size();
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Thread: " << this_id << ", In MaybeAdjustPolicyForHelperAddedNodes(), size of queue to process: " << my_queue_size;
  if(my_queue_size > 0){
    while(foo->queue_of_vector_of_nodes_from_helper_added_by_this_thread.size() > 0){    
      std::vector<Node*> vector_of_nodes_from_helper_added_by_this_thread = foo->queue_of_vector_of_nodes_from_helper_added_by_this_thread.front();
      foo->queue_of_vector_of_nodes_from_helper_added_by_this_thread.pop();

      float branching_factor = 1.6f;
      int starting_depth_of_PV = foo->starting_depth_of_PVs_.front();
      foo->starting_depth_of_PVs_.pop();
      int amount_of_support = foo->amount_of_support_for_PVs_.front();
      foo->amount_of_support_for_PVs_.pop();

      // If amount_of_support is zero, then this is a quiscence search, and in that case do no policy adjustment
      if(amount_of_support == 0){
	continue;
      }

      // if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Thread: " << this_id << ", In MaybeAdjustPolicyForHelperAddedNodes(), successfully read starting depth and amount of support.";

      // Do we want to maximize or minimize Q? At root, and thus at even depth, we want to _minimize_ Q (Q is from the perspective of the player who _made the move_ leading up the current position. Calculate depth at the first added node.
      int depth = 0;
      search_->nodes_mutex_.lock_shared();
      if(vector_of_nodes_from_helper_added_by_this_thread[0]->IsTerminal()){
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Thread: " << this_id << ", In MaybeAdjustPolicyForHelperAddedNodes(). Node is terminal, no need to modify its policy.";
	search_->nodes_mutex_.unlock_shared();	
	continue;
      }
      // LOGFILE << "Starting to calculate depth for this node: " << vector_of_nodes_from_helper_added_by_this_thread[0]->DebugString();
      for (Node* n2 = vector_of_nodes_from_helper_added_by_this_thread[0]; n2 != search_->root_node_ && depth < 100; n2 = n2->GetParent()) {
	depth++;
	if(n2->GetParent() == nullptr) {
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Thread: " << this_id << ", In MaybeAdjustPolicyForHelperAddedNodes(), Problem in depth calculation.";
	  break;
	}
      }
      search_->nodes_mutex_.unlock_shared();
      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "How good is this line I just added based on recommendations from the helper?";
      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Q of parent to the first added node in the line: " << vector_of_nodes_from_helper_added_by_this_thread[0]->GetParent()->GetQ(0.0f) << " depth: " << depth - 1;
      for(long unsigned int j = 0; j < vector_of_nodes_from_helper_added_by_this_thread.size(); j++){
	Node* n = vector_of_nodes_from_helper_added_by_this_thread[j];

	// divide the amount of support with the current depth ^ scaling factor to the ge current support
	int current_depth = depth - starting_depth_of_PV + j;
	int current_amount_of_support = float(amount_of_support) / pow(current_depth, branching_factor);
	if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "MaybeAdjustPolicyForHelperAddedNodes() at a node with current depth (distance from first node in PV) = " << current_depth << ". Starting depth of PV: " << starting_depth_of_PV << ". Distance from root for current node: " << depth+current_depth << ", amount of support for the PV: " << amount_of_support << " amount of support for this node: " << current_amount_of_support;

	// Strategies for policy adjustment:
	// a "trust the helper", make sure policy is at least c
	// b "only trust yourself, even with deep analysis", make sure policy is at least c when the move is promising.
	// c "only trust yourself, don't trust deep analysis", when the move is promising, make sure policy is at least x, which varies with depth.
	// d "let Q speak", set policy to equal to the policy of the sibling with highest policy.
	// e like d, but set policy slightly higher than the policy of the sibling with highest policy if the move is promising.
	// e is the current strategy
	
	std::string strategy;
	float current_p;
	float c = 0.175f;
	float d = 0.225f;
	float min_c = 0.0f;
	float minimum_policy = min_c;
	strategy = "e";

	if(strategy == "a") minimum_policy = c;

	if(strategy == "d" || strategy == "e"){
	  // make sure that policy is at least as good as the best sibling.
	  float highest_p = 0;
	  // loop through the policies of the siblings.
	  search_->nodes_mutex_.lock_shared();
	  for (auto& edge : n->GetParent()->Edges()) {
	    if(edge.GetP() > highest_p) highest_p = edge.GetP();
	  }
	  // While we have a lock, also get policy of the current nodw.
	  current_p = n->GetOwnEdge()->GetP();
	  search_->nodes_mutex_.unlock_shared();	  
	  minimum_policy = highest_p;
	}

	// boost nodes with a lot of support
	if(current_amount_of_support > 100000) minimum_policy = std::min(0.90, minimum_policy * 1.1);	    

	if(strategy != "a"){

	  // Determine if the move is promising or not
	  // if starting node is maximising and we are maximising: are we greater?
	  // if starting node is maximising and we are minimizing: are (-we) greater?
	  // if starting node is minimizing and we are maximising: are we greater than (-start node)?
	  // if starting node is minimizing and we are minimizing: are (-we) greater than (-start node)?
	  // we are maximising if depth + j % 2 == 1
	  // startnode is maximising if depth % 2 == 1
	  // signed int factor_for_us = ((depth + j) % 2 == 1) ? 1 : -1;
	  // signed int factor_for_parent = ((depth - 1) % 2 == 1) ? 1 : -1;
	  // Change, compare current node only with its parent
	  signed int factor_for_us = ((depth + j) % 2 == 1) ? 1 : -1;
	  signed int factor_for_parent = factor_for_us * -1;
	  // signed int factor_for_root = -1;

	  search_->nodes_mutex_.lock_shared();
	  // float root_q = factor_for_root * search_->root_node_->GetQ(0.0f); 
	  
	  if(factor_for_us * n->GetQ(0.0f) > factor_for_parent * n->GetParent()->GetQ(0.0f)){
	    if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "(Raw Q=" << n->GetQ(0.0f) << ") " << factor_for_us * n->GetQ(0.0f) << " is greater than " << factor_for_parent * n->GetParent()->GetQ(0.0f) << " which means this is promising. P: " << n->GetOwnEdge()->GetP() << " N: " << n->GetN() << " depth: " << depth + j;	    
	    // the move is promising
	    if(strategy == "b") minimum_policy = d;
	    if(strategy == "e") minimum_policy = std::min(0.90, minimum_policy * 1.2);

	    // If the move is better than root, then enqueue it
	    // if(factor_for_us * n->GetQ(0.0f) > root_q){
	    // LOGFILE << "We are better than root. we: " << factor_for_us * n->GetQ(0.0f) << " root: " << root_q << " will enqueue it (unless the node is terminal).";
	    // Check that it's not terminal and not already queued and not too deep.
	    if(!n->IsTerminal() && n->GetAuxEngineMove() == 0xffff && depth < 35 && current_amount_of_support > 5000){
	      AuxMaybeEnqueueNode(n);
	    }
	    
	  } else {
	    // Not promising, but since the helper recommended it, it is probably better than its policy, so give it some policy boosting.
	    if(strategy == "e") minimum_policy = std::min(0.90, minimum_policy * 0.85);	    
	  }
	  search_->nodes_mutex_.unlock_shared();	  
	}

	// if(strategy == "c"){
	//   // This is still under construction.
	//   // divide the amount of support with the current depth ^ scaling factor to the ge current support
	//   int current_depth = depth - starting_depth_of_PV + j;
	//   int current_amount_of_support = float(amount_of_support) / pow(current_depth, branching_factor);
	//   LOGFILE << "MaybeAdjustPolicyForHelperAddedNodes() at a node with current depth (distance from first node in PV) = " << current_depth << ". Starting depth of PV: " << starting_depth_of_PV << ". Distance from root for current node: " << depth << ", number of added nodes: " << my_pv_size << ", amount of support for the PV: " << amount_of_support << " amount of support for this node: " << current_amount_of_support;
	//   minimum_policy = std::max(min_c, (1.0f - float(j)/float(my_pv_size)) * c);
	// }

	// Actually adjust the policy to minimum_policy (if it is not already higher than that).
	if(current_p < minimum_policy){
	  float scaling_factor = 1/(1 + minimum_policy - current_p);
	  // First increase the policy to the desired value, then scale down policy of all nodes
	  search_->nodes_mutex_.lock();
	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Aquired a write lock on nodes to change policy";
	  n->GetOwnEdge()->SetP(minimum_policy);
	  // Now scale all policies down with the scaling factor.
	  for (const auto& child : n->GetParent()->Edges()) {
	    auto* edge = child.edge();
	    edge->SetP(edge->GetP() * scaling_factor);
	  }
	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Increased policy from " << current_p << " to " << minimum_policy * scaling_factor << " and scaled all other policies down by " << scaling_factor << " since the node was promising, now releasing the lock.";
	  search_->nodes_mutex_.unlock();
	  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Successfully released the lock.";
	}
      }
      // That's the new nodes, but what about the already existing nodes, shouldn't we boost policy for those too? (if they are promising)
      search_->nodes_mutex_.lock_shared();      
      for (Node* n2 = vector_of_nodes_from_helper_added_by_this_thread[0]; depth > 0; n2 = n2->GetParent()) {

	// If n2 has no visited siblings, skip all of this
	int children = 0;
	for (Node* child : n2->GetParent()->VisitedNodes()) {
	  if(child->GetN() > 0){ // unnecessary, but avoids a warning about unused variable child.
	    children++;
	  }
	}
	if(children > 1){

	  // // Make sure that policy is at least as good as the best sibling if no other node has a higher Q.
	  // float max_allowed_policy_for_non_best_child = 0.2f;

	  // If there is another node with higher Q, then reduce P of this node unless P is already not highest and not above 0.2
	  // If the highest policy is less than 0.2, then allow this node to get a P lower than 0.2, so that the other sibling can get highest P eventually.
	
	  float highest_p = 0;
	  Node * this_node_has_highest_p;	  
	  // loop through the policies of the siblings.
	  for (auto& edge : n2->GetParent()->Edges()) {
	    if(edge.GetP() > highest_p) {
	      highest_p = edge.GetP();
	      this_node_has_highest_p = edge.node();
	    }
	  }
	  // float minimum_policy = highest_p;

	  float highest_q = -1;
	  Node * this_node_has_best_q;
	  
	  for (Node* child : n2->GetParent()->VisitedNodes()) {
	    if(child->GetQ(0.5) > highest_q){
	      // LOGFILE << "At depth: " << depth << " " << child->GetQ(0.5) << " is greater than or equal to " << highest_q << " so this child is promising" << child->DebugString();
	      highest_q = child->GetQ(0.5);
	      this_node_has_best_q = child;
	    }
	  }
	  // Regardless of our node, also adjust the nodes with highest p and highest q, if needed.
	  if(this_node_has_best_q != nullptr && this_node_has_highest_p != nullptr && this_node_has_best_q != this_node_has_highest_p){
	    // increase p on this_node_has_best_q and decrease it on highest_p.
	    float diff_to_apply = this_node_has_highest_p->GetOwnEdge()->GetP() * 0.05;
	    // LOGFILE << "Changing policy on node: " << this_node_has_highest_p->DebugString() << " which has the highest policy, from " << this_node_has_highest_p->GetOwnEdge()->GetP() << " to " << this_node_has_highest_p->GetOwnEdge()->GetP()-diff_to_apply << " and policy on node: " << this_node_has_best_q->DebugString() << " which has policy " << this_node_has_best_q->GetOwnEdge()->GetP();
	    // upgrade the lock.
	    search_->nodes_mutex_.unlock_shared();
	    search_->nodes_mutex_.lock();
	    this_node_has_highest_p->GetOwnEdge()->SetP(this_node_has_highest_p->GetOwnEdge()->GetP()-diff_to_apply);
	    this_node_has_best_q->GetOwnEdge()->SetP(this_node_has_best_q->GetOwnEdge()->GetP()+diff_to_apply);
	    // downgrade lock
	    search_->nodes_mutex_.unlock();
	    search_->nodes_mutex_.lock_shared();

	    // Enqueue it, if it has not already been checked or is too deep.
	    if(!this_node_has_best_q->IsTerminal() && this_node_has_best_q->GetAuxEngineMove() == 0xffff && depth < 30){
	      AuxMaybeEnqueueNode(this_node_has_best_q);
	    }
	    
	  }
	} // End of children > 1
	depth--;
      } // End of policy boosting for existing nodes.
      search_->nodes_mutex_.unlock_shared();
      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "MaybeAdjustPolicy released a lock on nodes.";    
    }
    // Reset the variable, if it was non-empty.
    foo->queue_of_vector_of_nodes_from_helper_added_by_this_thread = {};
  }
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "MaybeAdjustPolicyForHelperAddedNodes() finished";
}

  

// 7. Update the Search's status and progress information.
//~~~~~~~~~~~~~~~~~~~~
void SearchWorker::UpdateCounters() {
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_, true);
  search_->MaybeOutputInfo();

  // If this thread had no work, not even out of order, then sleep for some
  // milliseconds. Collisions don't count as work, so have to enumerate to find
  // out if there was anything done.
  bool work_done = number_out_of_order_ > 0;
  if (!work_done) {
    for (NodeToProcess& node_to_process : minibatch_) {
      if (!node_to_process.IsCollision()) {
        work_done = true;
        break;
      }
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace lczero
