/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

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

#include "selfplay/multigame.h"

namespace lczero {

class PolicyEvaluator : public Evaluator {
 public:
  void Reset(const PlayerOptions& player) override {
    comp = player.network->NewComputation();
    input_format = player.network->GetCapabilities().input_format;
    transforms.clear();
    comp_idx = 0;
  }
  void Gather(NodeTree* tree) override {
    int transform;
    auto planes =
        EncodePositionForNN(input_format, tree->GetPositionHistory(), 8,
                            FillEmptyHistory::FEN_ONLY, &transform);
    transforms.push_back(transform);
    comp->AddInput(std::move(planes));
  }
  void Run() override { comp->ComputeBlocking(); }
  void MakeBestMove(NodeTree* tree) override {
    Move best;
    float max_p = std::numeric_limits<float>::lowest();
    for (auto edge : tree->GetCurrentHead()->Edges()) {
      float p = comp->GetPVal(comp_idx,
                              edge.GetMove().as_nn_index(transforms[comp_idx]));
      if (p >= max_p) {
        max_p = p;
        best = edge.GetMove(tree->GetPositionHistory().IsBlackToMove());
      }
    }
    tree->MakeMove(best);
    comp_idx++;
  }

  std::unique_ptr<NetworkComputation> comp;
  pblczero::NetworkFormat::InputFormat input_format;
  int comp_idx;
  std::vector<int> transforms;
};

class ValueEvaluator : public Evaluator {
 public:
  void Reset(const PlayerOptions& player) override {
    comp = player.network->NewComputation();
    input_format = player.network->GetCapabilities().input_format;
    comp_idx = 0;
  }
  void Gather(NodeTree* tree) override {
    PositionHistory history = tree->GetPositionHistory();
    for (auto edge : tree->GetCurrentHead()->Edges()) {
      history.Append(edge.GetMove());
      if (history.ComputeGameResult() == GameResult::UNDECIDED) {
        int transform;
        auto planes = EncodePositionForNN(
            input_format, history, 8, FillEmptyHistory::FEN_ONLY, &transform);
        comp->AddInput(std::move(planes));
      }
      history.Pop();
    }
  }
  void Run() override { comp->ComputeBlocking(); }
  void MakeBestMove(NodeTree* tree) override {
    Move best;
    float max_q = std::numeric_limits<float>::lowest();
    PositionHistory history = tree->GetPositionHistory();
    for (auto edge : tree->GetCurrentHead()->Edges()) {
      history.Append(edge.GetMove());
      auto result = history.ComputeGameResult();
      float q = -1;
      if (result == GameResult::UNDECIDED) {
        // NN eval is for side to move perspective - so if its good, its bad for
        // us.
        q = -comp->GetQVal(comp_idx);
        comp_idx++;
      } else if (result == GameResult::DRAW) {
        q = 0;
      } else {
	LOGFILE << "in MakeBestMove, found a decisive result = " << static_cast<int>(result);      
        // A legal move to a non-drawn terminal without tablebases must be a
        // win.
	//   q = 1;
	// this assumes win => q == 1 which is not true for all wins in Rmobility.
	// instead use a transformed version rmobility score as q 
	// This is copy/paste from trainingdata.cc,
	if (result == GameResult::WHITE_WON || result == GameResult::BLACK_WON) {
	  q = 1;
	} else if (result == GameResult::WHITE_STALEMATE || result == GameResult::BLACK_STALEMATE) {
	  // for r-mobility the points in https://wiki.chessdom.org/R-Mobility#50-move_rule must be scaled to the range of q [-1, 1], which means multiply by 2, and then subtract 1
	  q = 0.5; // 0.75 * 2 - 1 = 0.5
	} else if (result == GameResult::WHITE_G1_0 || result == GameResult::BLACK_G1_0) {
	  q = 0.25;
	} else if (result == GameResult::WHITE_G1_5 || result == GameResult::BLACK_G1_5) {
	  q = 0.125;
	} else if (result == GameResult::WHITE_G2_0 || result == GameResult::BLACK_G2_0) {
	  q = 0.0625;
	} else if (result == GameResult::WHITE_G2_5 || result == GameResult::BLACK_G2_5) {
	  q = 0.03125;
	} else if (result == GameResult::WHITE_G3_0 || result == GameResult::BLACK_G3_0) {
	  q = 0.015625;
	} else if (result == GameResult::WHITE_G3_5 || result == GameResult::BLACK_G3_5) {
	  q = 0.0078125;
	} else if (result == GameResult::WHITE_G4_0 || result == GameResult::BLACK_G4_0) {
	  q = 0.00390625;
	} else if (result == GameResult::WHITE_G4_5 || result == GameResult::BLACK_G4_5) {
	  q = 0.001953125;
	} else if (result == GameResult::WHITE_G5_0 || result == GameResult::BLACK_G5_0) {
	  q = 0.0009765625;
	} else if (result == GameResult::WHITE_G5_5 || result == GameResult::BLACK_G5_5) {
	  q = 0.0004882812;
	} else if (result == GameResult::WHITE_G6_0 || result == GameResult::BLACK_G6_0) {
	  q = 0.0002441406;
	} else if (result == GameResult::WHITE_G6_5 || result == GameResult::BLACK_G6_5) {
	  q = 0.0001220703;
	} else if (result == GameResult::WHITE_G7_0 || result == GameResult::BLACK_G7_0) {
	  q = 0.00006103516;
	} else if (result == GameResult::WHITE_G7_5 || result == GameResult::BLACK_G7_5) {
	  q = 0.00003051758;
	} else if (result == GameResult::WHITE_G8_0 || result == GameResult::BLACK_G8_0) {
	  q = 0.00001525879;
	} else if (result == GameResult::WHITE_G8_5 || result == GameResult::BLACK_G8_5) {
	  q = 0.000007629395;
	} else if (result == GameResult::WHITE_G9_0 || result == GameResult::BLACK_G9_0) {
	  q = 0.000003814697;
	} else if (result == GameResult::WHITE_G9_5 || result == GameResult::BLACK_G9_5) {
	  q = 0.000001907349;
	}
      }
      if (q >= max_q) {
        max_q = q;
        best = edge.GetMove(tree->GetPositionHistory().IsBlackToMove());
      }
      history.Pop();
    }
    tree->MakeMove(best);
  }

  std::unique_ptr<NetworkComputation> comp;
  pblczero::NetworkFormat::InputFormat input_format;
  int comp_idx;
  std::vector<int> transforms;
};

MultiSelfPlayGames::MultiSelfPlayGames(PlayerOptions player1,
                                       PlayerOptions player2,
                                       const std::vector<Opening>& openings,
                                       SyzygyTablebase* syzygy_tb,
                                       bool use_value)
    : options_{player1, player2}, syzygy_tb_(syzygy_tb) {
  eval_ = use_value
              ? std::unique_ptr<Evaluator>(std::make_unique<ValueEvaluator>())
              : std::unique_ptr<Evaluator>(std::make_unique<PolicyEvaluator>());
  trees_.reserve(openings.size());
  for (auto opening : openings) {
    trees_.push_back(std::make_shared<NodeTree>());
    trees_.back()->ResetToPosition(opening.start_fen, {});
    results_.push_back(GameResult::UNDECIDED);

    for (Move m : opening.moves) {
      trees_.back()->MakeMove(m);
    }
  }
}

void MultiSelfPlayGames::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
}

void MultiSelfPlayGames::Play() {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
    }
    bool all_done = true;
    bool blacks_move = false;
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] == GameResult::UNDECIDED) {
        if (tree->GetPositionHistory().ComputeGameResult() !=
            GameResult::UNDECIDED) {
          results_[i] = tree->GetPositionHistory().ComputeGameResult();
          continue;
        }
        if (syzygy_tb_ != nullptr) {
          auto board = tree->GetPositionHistory().Last().GetBoard();
          if (board.castlings().no_legal_castle() &&
              (board.ours() | board.theirs()).count() <=
                  syzygy_tb_->max_cardinality()) {
            auto tb_side_black = (tree->GetPlyCount() % 2) == 1;
            ProbeState state;
            const WDLScore wdl = syzygy_tb_->probe_wdl(
                tree->GetPositionHistory().Last(), &state);
            // Only fail state means the WDL is wrong, probe_wdl may produce
            // correct result with a stat other than OK.
            if (state != FAIL) {
              if (wdl == WDL_WIN) {
                results_[i] = tb_side_black ? GameResult::BLACK_WON
                                            : GameResult::WHITE_WON;
              } else if (wdl == WDL_LOSS) {
                results_[i] = tb_side_black ? GameResult::WHITE_WON
                                            : GameResult::BLACK_WON;
              } else {  // Cursed wins and blessed losses count as draws.
		// Rmobility TODO, no TB win does not imply draw for Rmobility
		// do we *have* to set results_[i] or can we just not do it?
                results_[i] = GameResult::DRAW;
              }
              continue;
            }
          }
        }
        if (all_done) {
          all_done = false;
          blacks_move = (tree->GetPlyCount() % 2) == 1;
          // Don't break as we need to update result state for everything.
        }
      }
    }
    if (all_done) break;
    const int idx = blacks_move ? 1 : 0;
    eval_->Reset(options_[idx]);
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] != GameResult::UNDECIDED) {
        continue;
      }
      if (((tree->GetPlyCount() % 2) == 1) != blacks_move) continue;
      const auto& board = tree->GetPositionHistory().Last().GetBoard();
      auto legal_moves = board.GenerateLegalMoves();
      tree->GetCurrentHead()->CreateEdges(legal_moves);
      eval_->Gather(tree.get());
    }
    eval_->Run();
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] != GameResult::UNDECIDED) {
        continue;
      }
      if (((tree->GetPlyCount() % 2) == 1) != blacks_move) continue;
      eval_->MakeBestMove(tree.get());
    }
  }
}

}  // namespace lczero
