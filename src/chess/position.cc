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

#include "chess/position.h"

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include "utils/logging.h"

namespace {
// GetPieceAt returns the piece found at row, col on board or the null-char '\0'
// in case no piece there.
char GetPieceAt(const lczero::ChessBoard& board, int row, int col) {
  char c = '\0';
  if (board.ours().get(row, col) || board.theirs().get(row, col)) {
    if (board.pawns().get(row, col)) {
      c = 'P';
    } else if (board.kings().get(row, col)) {
      c = 'K';
    } else if (board.bishops().get(row, col)) {
      c = 'B';
    } else if (board.queens().get(row, col)) {
      c = 'Q';
    } else if (board.rooks().get(row, col)) {
      c = 'R';
    } else {
      c = 'N';
    }
    if (board.theirs().get(row, col)) {
      c = std::tolower(c);  // Capitals are for white.
    }
  }
  return c;
}

}  // namespace
namespace lczero {

Position::Position(const Position& parent, Move m)
    : rule50_ply_(parent.rule50_ply_ + 1), ply_count_(parent.ply_count_ + 1) {
  them_board_ = parent.us_board_;
  const bool is_zeroing = them_board_.ApplyMove(m);
  us_board_ = them_board_;
  us_board_.Mirror();
  if (is_zeroing) rule50_ply_ = 0;
}

Position::Position(const ChessBoard& board, int rule50_ply, int game_ply)
    : rule50_ply_(rule50_ply), repetitions_(0), ply_count_(game_ply) {
  us_board_ = board;
  them_board_ = board;
  them_board_.Mirror();
}

uint64_t Position::Hash() const {
  return HashCat({us_board_.Hash(), static_cast<unsigned long>(repetitions_)});
}

std::string Position::DebugString() const { return us_board_.DebugString(); }

GameResult operator-(const GameResult& res) {
  return res == GameResult::BLACK_WON   ? GameResult::WHITE_WON
         : res == GameResult::WHITE_WON ? GameResult::BLACK_WON
                                        : res;
}

GameResult PositionHistory::ComputeGameResultRmobility() const {
  // traverse the game history until the last move that reset the 50 ply move rule (pawn move or capture)
  // find out which side first reached the highest goal that was reached, and what that goal was.
  LOGFILE << "Calculating R mobility score. The value of rule50_ply_ for the previous position was " << Last().GetRule50Ply() << ", number of elements in history: " << GetLength();
  struct {
    long unsigned int number_of_legal_moves;
    bool is_in_check;
    bool white_is_best_player;
  } best_goal;
  best_goal.number_of_legal_moves = 10;
  best_goal.is_in_check = false;
  bool is_black_to_move = IsBlackToMove();
  GameResult result = GameResult::DRAW;
  uint8_t result_as_int;
  for(int i = 1; i <= Last().GetRule50Ply(); i++){
    // does the current position equal or beat the previous goal AND beat G10.0 which is best non-winning position?
    const auto& board = GetPositionAt(GetLength() - i - 1).GetBoard();
    auto legal_moves = board.GenerateLegalMoves();
    if(legal_moves.size() < 10 && legal_moves.size() <= best_goal.number_of_legal_moves){
      best_goal.number_of_legal_moves = legal_moves.size();
      best_goal.is_in_check = board.IsUnderCheck();
      best_goal.white_is_best_player = ! is_black_to_move;
      // This fits the order defined in position.h line 97
      result_as_int = 1 + is_black_to_move * 2 * legal_moves.size() + ! is_black_to_move * 20 + ! is_black_to_move * 2 * (9 - legal_moves.size()) + ! board.IsUnderCheck();
      LOGFILE << "result_as_int = " << +result_as_int;
      result = static_cast<GameResult>(result_as_int);
      if(best_goal.white_is_best_player){
	if(best_goal.is_in_check){
	  LOGFILE << "White reached a new highest goal. number of legal moves: " << best_goal.number_of_legal_moves << " and in check at ply: " << positions_.size() - i;
	} else {
	  LOGFILE << "White reached a new highest goal. number of legal moves: " << best_goal.number_of_legal_moves << " not in check at ply: " << positions_.size() - i;	  
	}
      } else {
	if(best_goal.is_in_check){
	  LOGFILE << "Black reached a new highest goal. number of legal moves: " << best_goal.number_of_legal_moves << " and in check at ply: " << positions_.size() - i;
	} else {
	  LOGFILE << "Black reached a new highest goal. number of legal moves: " << best_goal.number_of_legal_moves << " not in check at ply: " << positions_.size() - i;	  
	}
      }
    }
    // switch player for the next iteration
    is_black_to_move = !is_black_to_move;
  }
  // Log the result to make sure I got this right, remove when convinced.
  switch(result_as_int) {
  case 1:
    LOGFILE << "Result: black won by checkmate";
    break;
  case 2:
    LOGFILE << "Result: black won by stalemate";
    break;
  case 3:
    LOGFILE << "Result: black won by r-mobility G1.0";
    break;
  case 4:
    LOGFILE << "Result: black won by r-mobility G1.5";
    break;
  case 5:
    LOGFILE << "Result: black won by r-mobility G2.0";
    break;
  case 6:
    LOGFILE << "Result: black won by r-mobility G2.5";
    break;
  case 7:
    LOGFILE << "Result: black won by r-mobility G3.0";
    break;
  case 8:
    LOGFILE << "Result: black won by r-mobility G3.5";
    break;
  case 9:
    LOGFILE << "Result: black won by r-mobility G4.0";
    break;
  case 10:
    LOGFILE << "Result: black won by r-mobility G4.5";
    break;
  case 11:
    LOGFILE << "Result: black won by r-mobility G5.0";
    break;
  case 12:
    LOGFILE << "Result: black won by r-mobility G5.5";
    break;
  case 13:
    LOGFILE << "Result: black won by r-mobility G6.0";
    break;
  case 14:
    LOGFILE << "Result: black won by r-mobility G6.5";
    break;
  case 15:
    LOGFILE << "Result: black won by r-mobility G7.0";
    break;
  case 16:
    LOGFILE << "Result: black won by r-mobility G7.5";
    break;
  case 17:
    LOGFILE << "Result: black won by r-mobility G8.0";
    break;
  case 18:
    LOGFILE << "Result: black won by r-mobility G8.5";
    break;
  case 19:
    LOGFILE << "Result: black won by r-mobility G9.0";
    break;
  case 20:
    LOGFILE << "Result: black won by r-mobility G9.5";
    break;
  case 21:
    LOGFILE << "Result: draw";
    break;
  case 22:
    LOGFILE << "Result: white won by stalemate";
    break;
  case 24:
    LOGFILE << "Result: white won by r-mobility G1.0";
    break;
  case 25:
    LOGFILE << "Result: white won by r-mobility G1.5";
    break;
  case 26:
    LOGFILE << "Result: white won by r-mobility G2.0";
    break;
  case 27:
    LOGFILE << "Result: white won by r-mobility G2.5";
    break;
  case 28:
    LOGFILE << "Result: white won by r-mobility G3.0";
    break;
  case 29:
    LOGFILE << "Result: white won by r-mobility G3.5";
    break;
  case 30:
    LOGFILE << "Result: white won by r-mobility G4.0";
    break;
  case 31:
    LOGFILE << "Result: white won by r-mobility G4.5";
    break;
  case 32:
    LOGFILE << "Result: white won by r-mobility G5.0";
    break;
  case 33:
    LOGFILE << "Result: white won by r-mobility G5.5";
    break;
  case 34:
    LOGFILE << "Result: white won by r-mobility G6.0";
    break;
  case 35:
    LOGFILE << "Result: white won by r-mobility G6.5";
    break;
  case 36:
    LOGFILE << "Result: white won by r-mobility G7.0";
    break;
  case 37:
    LOGFILE << "Result: white won by r-mobility G7.5";
    break;
  case 38:
    LOGFILE << "Result: white won by r-mobility G8.0";
    break;
  case 39:
    LOGFILE << "Result: white won by r-mobility G8.5";
    break;
  case 40:
    LOGFILE << "Result: white won by r-mobility G9.0";
    break;
  case 41:
    LOGFILE << "Result: white won by r-mobility G9.5";
    break;
  }
  return result;
}

GameResult PositionHistory::ComputeGameResult() const {
  const auto& board = Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();
  if (legal_moves.empty()) {
    if (board.IsUnderCheck()) {
      // Checkmate.
      LOGFILE << "Result: won by checkmate";      
      return IsBlackToMove() ? GameResult::WHITE_WON : GameResult::BLACK_WON;
    }
    // Stalemate.
    LOGFILE << "Result: won by stalemate";
    return IsBlackToMove() ? GameResult::WHITE_STALEMATE : GameResult::BLACK_STALEMATE;
  }

  // if (!board.HasMatingMaterial()) return GameResult::DRAW;
  if (Last().GetRule50Ply() >= 100) {
    return ComputeGameResultRmobility();
  }
  // if (Last().GetRepetitions() >= 2) return GameResult::DRAW;
  if (Last().GetRepetitions() >= 2) {
    // LOGFILE << "Result: draw by repetitions";
    // return GameResult::DRAW;
    return ComputeGameResultRmobility();    
  }

  return GameResult::UNDECIDED;
}

void PositionHistory::Reset(const ChessBoard& board, int rule50_ply,
                            int game_ply) {
  positions_.clear();
  positions_.emplace_back(board, rule50_ply, game_ply);
}

void PositionHistory::Append(Move m) {
  // TODO(mooskagh) That should be emplace_back(Last(), m), but MSVS STL
  //                has a bug in implementation of emplace_back, when
  //                reallocation happens. (it also reallocates Last())
  positions_.push_back(Position(Last(), m));
  int cycle_length;
  int repetitions = ComputeLastMoveRepetitions(&cycle_length);
  positions_.back().SetRepetitions(repetitions, cycle_length);
}

int PositionHistory::ComputeLastMoveRepetitions(int* cycle_length) const {
  *cycle_length = 0;
  const auto& last = positions_.back();
  // TODO(crem) implement hash/cache based solution.
  if (last.GetRule50Ply() < 4) return 0;

  for (int idx = positions_.size() - 3; idx >= 0; idx -= 2) {
    const auto& pos = positions_[idx];
    if (pos.GetBoard() == last.GetBoard()) {
      *cycle_length = positions_.size() - 1 - idx;
      return 1 + pos.GetRepetitions();
    }
    if (pos.GetRule50Ply() < 2) return 0;
  }
  return 0;
}

bool PositionHistory::DidRepeatSinceLastZeroingMove() const {
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (iter->GetRepetitions() > 0) return true;
    if (iter->GetRule50Ply() == 0) return false;
  }
  return false;
}

uint64_t PositionHistory::HashLast(int positions) const {
  uint64_t hash = positions;
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (!positions--) break;
    hash = HashCat(hash, iter->Hash());
  }
  return HashCat(hash, Last().GetRule50Ply());
}

std::string GetFen(const Position& pos) {
  std::string result;
  const ChessBoard& board = pos.GetWhiteBoard();
  for (int row = 7; row >= 0; --row) {
    int emptycounter = 0;
    for (int col = 0; col < 8; ++col) {
      char piece = GetPieceAt(board, row, col);
      if (emptycounter > 0 && piece) {
        result += std::to_string(emptycounter);
        emptycounter = 0;
      }
      if (piece) {
        result += piece;
      } else {
        emptycounter++;
      }
    }
    if (emptycounter > 0) result += std::to_string(emptycounter);
    if (row > 0) result += "/";
  }
  std::string enpassant = "-";
  if (!board.en_passant().empty()) {
    auto sq = *board.en_passant().begin();
    enpassant = BoardSquare(pos.IsBlackToMove() ? 2 : 5, sq.col()).as_string();
  }
  result += pos.IsBlackToMove() ? " b" : " w";
  result += " " + board.castlings().as_string();
  result += " " + enpassant;
  result += " " + std::to_string(pos.GetRule50Ply());
  result += " " + std::to_string(
                      (pos.GetGamePly() + (pos.IsBlackToMove() ? 1 : 2)) / 2);
  return result;
}
}  // namespace lczero
