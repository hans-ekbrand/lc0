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

#include "selfplay/loop.h"

#include <optional>
#include <sstream>

#include "gtb-probe.h"
#include "neural/decoder.h"
#include "selfplay/tournament.h"
#include "trainingdata/reader.h"
#include "utils/configfile.h"
#include "utils/filesystem.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {

namespace {
const OptionId kInteractiveId{
    "interactive", "", "Run in interactive mode with UCI-like interface."};
const OptionId kSyzygyTablebaseId{"syzygy-paths", "",
                                  "List of Syzygy tablebase directories"};
const OptionId kGaviotaTablebaseId{"gaviotatb-paths", "",
                                   "List of Gaviota tablebase directories"};
const OptionId kInputDirId{
    "input", "", "Directory with gzipped files in need of rescoring."};
const OptionId kPolicySubsDirId{"policy-substitutions", "",
                                "Directory with gzipped files are to use to "
                                "replace policy for some of the data."};
const OptionId kOutputDirId{"output", "", "Directory to write rescored files."};
const OptionId kThreadsId{"threads", "",
                          "Number of concurrent threads to rescore with.", 't'};
const OptionId kTempId{"temperature", "",
                       "Additional temperature to apply to policy target."};
const OptionId kDistributionOffsetId{
    "dist_offset", "",
    "Additional offset to apply to policy target before temperature."};
const OptionId kMinDTZBoostId{
    "dtz_policy_boost", "",
    "Additional offset to apply to policy target before temperature for moves "
    "that are best dtz option."};
const OptionId kNewInputFormatId{
    "new-input-format", "",
    "Input format to convert training data to during rescoring."};
const OptionId kDeblunder{
    "deblunder", "",
    "If true, whether to use move Q information to infer a different Z value "
    "if the the selected move appears to be a blunder."};
const OptionId kDeblunderQBlunderThreshold{
    "deblunder-q-blunder-threshold", "",
    "The amount Q of played move needs to be worse than best move in order to "
    "assume the played move is a blunder."};
const OptionId kDeblunderQBlunderWidth{
    "deblunder-q-blunder-width", "",
    "Width of the transition between accepted temp moves and blunders."};
const OptionId kNnuePlainFileId{"nnue-plain-file", "",
                                "Append SF plain format training data to this "
                                "file. Will be generated if not there."};
const OptionId kNnueBestScoreId{"nnue-best-score", "",
                                "For the SF training data use the score of the "
                                "best move instead of the played one."};
const OptionId kNnueBestMoveId{
    "nnue-best-move", "",
    "For the SF training data record the best move instead of the played one. "
    "If set to true the generated files do not compress well."};
const OptionId kNumberOfPiecesId{"number-of-pieces", "",
    "Extract one position per game with this many pieces, it the game reached "
    "that state. See also delta-q."};
const OptionId kDeltaQId{"delta-q", "",
    "Only extract a position from the game if a position also satisfied the "
    "requirement that abs(root_q) is less than delta-q away from 0.5."};
const OptionId kDeleteFilesId{"delete-files", "",
                              "Delete the input files after processing."};
const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console."};

class PolicySubNode {
 public:
  PolicySubNode() {
    for (int i = 0; i < 1858; i++) children[i] = nullptr;
  }
  bool active = false;
  float policy[1858];
  PolicySubNode* children[1858];
};

std::atomic<int> games(0);
std::atomic<int> positions(0);
std::atomic<int> rescored(0);
std::atomic<int> delta(0);
std::atomic<int> rescored2(0);
std::atomic<int> rescored3(0);
std::atomic<int> blunders(0);
std::atomic<int> orig_counts[3];
std::atomic<int> fixed_counts[3];
std::atomic<int> policy_bump(0);
std::atomic<int> policy_nobump_total_hist[11];
std::atomic<int> policy_bump_total_hist[11];
std::atomic<int> policy_dtm_bump(0);
std::atomic<int> gaviota_dtm_rescores(0);
std::map<uint64_t, PolicySubNode> policy_subs;
bool gaviotaEnabled = false;
bool deblunderEnabled = false;
float deblunderQBlunderThreshold = 2.0f;
float deblunderQBlunderWidth = 0.0f;

void DataAssert(bool check_result) {
  if (!check_result) throw Exception("Range Violation");
}

void Validate(const std::vector<V7TrainingData>& fileContents) {
  if (fileContents.empty()) throw Exception("Empty File");

  for (long unsigned int i = 0; i < fileContents.size(); i++) {
    auto& data = fileContents[i];
    DataAssert(
        data.input_format ==
            pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION ||
        data.input_format == pblczero::NetworkFormat::
                                 INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
        data.input_format ==
            pblczero::NetworkFormat::
                INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
        data.input_format == pblczero::NetworkFormat::
                                 INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON);
    DataAssert(data.best_d >= 0.0f && data.best_d <= 1.0f);
    DataAssert(data.root_d >= 0.0f && data.root_d <= 1.0f);
    DataAssert(data.best_q >= -1.0f && data.best_q <= 1.0f);
    DataAssert(data.root_q >= -1.0f && data.root_q <= 1.0f);
    // std::cout << "root_m: " << data.root_m << " best_m: " << data.best_m << " plies left: " << data.plies_left << " \n";
    DataAssert(data.root_m >= 0.0f);
    DataAssert(data.best_m >= 0.0f);
    DataAssert(data.plies_left >= 0.0f);
    switch (data.input_format) {
      case pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE:
        DataAssert(data.castling_them_oo >= 0 && data.castling_them_oo <= 1);
        DataAssert(data.castling_them_ooo >= 0 && data.castling_them_ooo <= 1);
        DataAssert(data.castling_us_oo >= 0 && data.castling_us_oo <= 1);
        DataAssert(data.castling_us_ooo >= 0 && data.castling_us_ooo <= 1);
        break;
      default:
        // Verifiy at most one bit set.
        DataAssert((data.castling_them_oo & (data.castling_them_oo - 1)) == 0);
        DataAssert((data.castling_them_ooo & (data.castling_them_ooo - 1)) ==
                   0);
        DataAssert((data.castling_us_oo & (data.castling_us_oo - 1)) == 0);
        DataAssert((data.castling_us_ooo & (data.castling_us_ooo - 1)) == 0);
    }
    if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(
            data.input_format))) {
      // At most one en-passant bit.
      DataAssert((data.side_to_move_or_enpassant &
                  (data.side_to_move_or_enpassant - 1)) == 0);
    } else {
      DataAssert(data.side_to_move_or_enpassant >= 0 &&
                 data.side_to_move_or_enpassant <= 1);
    }
    DataAssert(data.result_q >= -1 && data.result_q <= 1);
    DataAssert(data.result_d >= 0 && data.result_q <= 1);
    DataAssert(data.rule50_count >= 0 && data.rule50_count <= 100);
    float sum = 0.0f;
    for (long unsigned int j = 0; j < sizeof(data.probabilities) / sizeof(float); j++) {
      float prob = data.probabilities[j];
      DataAssert((prob >= 0.0f && prob <= 1.0f) || prob == -1.0f ||
                 std::isnan(prob));
      if (prob >= 0.0f) {
        sum += prob;
      }
      // Only check best_idx/played_idx for real v6 data.
      if (data.visits > 0) {
        // Best_idx and played_idx must be marked legal in probabilities.
        if (j == data.best_idx || j == data.played_idx) {
          DataAssert(prob >= 0.0f);
        }
      }
    }
    if (sum < 0.99f || sum > 1.01f) {
      throw Exception("Probability sum error is huge!");
    }
    DataAssert(data.best_idx <= 1858);
    DataAssert(data.played_idx <= 1858);
    DataAssert(data.played_q >= -1.0f && data.played_q <= 1.0f);
    DataAssert(data.played_d >= 0.0f && data.played_d <= 1.0f);
    DataAssert(data.played_m >= 0.0f);
    DataAssert(std::isnan(data.orig_q) ||
               ((data.orig_q >= -1.0f) && (data.orig_q <= 1.0f)));
    DataAssert(std::isnan(data.orig_d) ||
               ((data.orig_d >= 0.0f) && (data.orig_d <= 1.0f)));
    DataAssert(std::isnan(data.orig_m) || data.orig_m >= 0.0f);
    // TODO: if visits > 0 - assert best_idx/played_idx are valid in
    // probabilities.
  }
}

void Validate(const std::vector<V7TrainingData>& fileContents,
              const MoveList& moves) {
  PositionHistory history;
  int rule50ply;
  int gameply;
  ChessBoard board;
  auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
      fileContents[0].input_format);
  PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]), &board,
                &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);
  for (long unsigned int i = 0; i < moves.size(); i++) {
    int transform = TransformForPosition(input_format, history);
    // If real v6 data, can confirm that played_idx matches the inferred move.
    if (fileContents[i].visits > 0) {
      if (fileContents[i].played_idx != moves[i].as_nn_index(transform)) {
	std::cout << "mismatch between moves and fileContents given to Validate() at position " << i << " move played: " << fileContents[i].played_idx << " move in movelist: " << moves[i].as_nn_index(transform) << "\n";
        throw Exception("Move performed is not listed as played.");
      }
    }
    // Move shouldn't be marked illegal unless there is 0 visits, which should
    // only happen if invariance_info is marked with the placeholder bit.
    if (!(fileContents[i].probabilities[moves[i].as_nn_index(transform)] >=
          0.0f) &&
        (fileContents[i].invariance_info & 64) == 0) {
      std::cerr << "Illegal move: " << moves[i].as_string() << std::endl;
      throw Exception("Move performed is marked illegal in probabilities.");
    }
    auto legal = history.Last().GetBoard().GenerateLegalMoves();
    if (std::find(legal.begin(), legal.end(), moves[i]) == legal.end()) {
      std::cerr << "Illegal move: " << moves[i].as_string() << std::endl;
      throw Exception("Move performed is an illegal move.");
    }
    history.Append(moves[i]);
  }
}

struct ProcessFileFlags {
  bool delete_files : 1;
  bool nnue_best_score : 1;
  bool nnue_best_move : 1;
};

void ProcessFile(const std::string& file, SyzygyTablebase* tablebase,
                 std::string outputDir, float delta_q, int number_of_pieces,
                 float dtzBoost, int newInputFormat,
                 std::string nnue_plain_file, ProcessFileFlags flags) {
  // Scope to ensure reader and writer are closed before deleting source file.
  {
    try {
      TrainingDataReader reader(file);
      std::vector<V7TrainingData> fileContents;
      V7TrainingData data;
      while (reader.ReadChunk(&data)) {
        fileContents.push_back(data);
      }
      // std::cout << "number of chunks (moves) found: " << fileContents.size() << "\n";
      Validate(fileContents);
      // std::cout << "First check passed \n";      
      MoveList moves;
      for (long unsigned int i = 1; i < fileContents.size(); i++) {
        moves.push_back(
            DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]),
                                PlanesFromTrainingData(fileContents[i - 1])));
        // All moves decoded are from the point of view of the side after the
        // move so need to mirror them all to be applicable to apply to the
        // position before.
        moves.back().Mirror();
	// std::cout << "Stored the move played in position " << i - 1 << ", which is the difference between position " << i - 1 << " and position " << i << " and the move was: " << moves.back().as_string() << ".\n";
	// The last move can not be checked this way since it has no position after it, since the final position can not be trained on.
      }
      Validate(fileContents, moves);
      games += 1;
      positions += fileContents.size();
      PositionHistory history;
      int rule50ply;
      int gameply;
      ChessBoard board;
      auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
          fileContents[0].input_format);
      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      uint64_t rootHash = HashCat(board.Hash(), rule50ply);

      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
		    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      for (long unsigned int i = 0; i < moves.size(); i++) {
	history.Append(moves[i]);
	const auto& board = history.Last().GetBoard();
	if((std::abs(fileContents[i].root_q) < 0.1) && (fileContents[i].root_d < 0.3)) {

	  std::string fen = GetFen(history.Last());
	  std::cout << "[FEN \"" << fen << "\"]\n\n\n";
	  std::cout << " draw: " << fileContents[i].root_d << " root_q: " << fileContents[i].root_q
		    << "\n";
	  break; // Only extract one position per game
	}
      }

    } catch (Exception& ex) {
      std::cerr << "While processing: " << file
                << " - Exception thrown: " << ex.what() << std::endl;
      if (flags.delete_files) {
        std::cerr << "It will be deleted." << std::endl;
      }
    }
  }
  if (flags.delete_files) {
    remove(file.c_str());
  }
}

void ProcessFiles(const std::vector<std::string>& files,
                  SyzygyTablebase* tablebase, std::string outputDir,
                  float delta_q, int number_of_pieces, float dtzBoost,
                  int newInputFormat, int offset, int mod,
                  std::string nnue_plain_file, ProcessFileFlags flags) {
  // std::cerr << "Thread: " << offset << " starting" << std::endl;
  for (long unsigned int i = offset; i < files.size(); i += mod) {
    if (files[i].rfind(".gz") != files[i].size() - 3) {
      std::cerr << "Skipping: " << files[i] << std::endl;
      continue;
    }
    ProcessFile(files[i], tablebase, outputDir, delta_q, number_of_pieces, dtzBoost,
                newInputFormat, nnue_plain_file, flags);
  }
}

void BuildSubs(const std::vector<std::string>& files) {
  for (auto& file : files) {
    TrainingDataReader reader(file);
    std::vector<V7TrainingData> fileContents;
    V7TrainingData data;
    while (reader.ReadChunk(&data)) {
      fileContents.push_back(data);
    }
    Validate(fileContents);
    MoveList moves;
    for (long unsigned int i = 1; i < fileContents.size(); i++) {
      moves.push_back(
          DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]),
                              PlanesFromTrainingData(fileContents[i - 1])));
      // All moves decoded are from the point of view of the side after the
      // move so need to mirror them all to be applicable to apply to the
      // position before.
      moves.back().Mirror();
    }
    Validate(fileContents, moves);

    // Subs are 'valid'.
    PositionHistory history;
    int rule50ply;
    int gameply;
    ChessBoard board;
    auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
        fileContents[0].input_format);
    PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]), &board,
                  &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);
    uint64_t rootHash = HashCat(board.Hash(), rule50ply);
    PolicySubNode* rootNode = &policy_subs[rootHash];
    for (long unsigned int i = 0; i < fileContents.size(); i++) {
      if ((fileContents[i].invariance_info & 64) == 0) {
        rootNode->active = true;
        for (int j = 0; j < 1858; j++) {
          rootNode->policy[j] = fileContents[i].probabilities[j];
        }
      }
      if (i < fileContents.size() - 1) {
        int transform = TransformForPosition(input_format, history);
        int idx = moves[i].as_nn_index(transform);
        if (rootNode->children[idx] == nullptr) {
          rootNode->children[idx] = new PolicySubNode();
        }
        rootNode = rootNode->children[idx];
        history.Append(moves[i]);
      }
    }
  }
}

}  // namespace

RescoreLoop::RescoreLoop() {}

RescoreLoop::~RescoreLoop() {}

#ifdef _WIN32
#define SEP_CHAR ';'
#else
#define SEP_CHAR ':'
#endif

void RescoreLoop::RunLoop() {
  orig_counts[0] = 0;
  orig_counts[1] = 0;
  orig_counts[2] = 0;
  fixed_counts[0] = 0;
  fixed_counts[1] = 0;
  fixed_counts[2] = 0;
  for (int i = 0; i < 11; i++) policy_bump_total_hist[i] = 0;
  for (int i = 0; i < 11; i++) policy_nobump_total_hist[i] = 0;
  options_.Add<StringOption>(kSyzygyTablebaseId);
  options_.Add<StringOption>(kGaviotaTablebaseId);
  options_.Add<StringOption>(kInputDirId);
  options_.Add<StringOption>(kOutputDirId);
  options_.Add<StringOption>(kPolicySubsDirId);
  options_.Add<IntOption>(kThreadsId, 1, 128) = 1;
  options_.Add<FloatOption>(kTempId, 0.001, 100) = 1;
  // Positive dist offset requires knowing the legal move set, so not supported
  // for now.
  options_.Add<FloatOption>(kDistributionOffsetId, -0.999, 0) = 0;
  options_.Add<FloatOption>(kMinDTZBoostId, 0, 1) = 0;
  options_.Add<IntOption>(kNewInputFormatId, -1, 256) = -1;
  options_.Add<BoolOption>(kDeblunder) = false;
  options_.Add<FloatOption>(kDeblunderQBlunderThreshold, 0.0f, 2.0f) = 2.0f;
  options_.Add<FloatOption>(kDeblunderQBlunderWidth, 0.0f, 2.0f) = 0.0f;
  options_.Add<StringOption>(kNnuePlainFileId);
  options_.Add<BoolOption>(kNnueBestScoreId) = true;
  options_.Add<BoolOption>(kNnueBestMoveId) = false;
  options_.Add<BoolOption>(kDeleteFilesId) = false;
  options_.Add<IntOption>(kNumberOfPiecesId, 2, 32) = 6;
  options_.Add<FloatOption>(kDeltaQId, 0.001, 0.4) = 0.1;

  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;

  if (options_.GetOptionsDict().IsDefault<std::string>(kOutputDirId) &&
      options_.GetOptionsDict().IsDefault<std::string>(kNnuePlainFileId)) {
    std::cerr << "Must provide an output dir or NNUE plain file." << std::endl;
    return;
  }

  deblunderEnabled = options_.GetOptionsDict().Get<bool>(kDeblunder);
  deblunderQBlunderThreshold =
      options_.GetOptionsDict().Get<float>(kDeblunderQBlunderThreshold);
  deblunderQBlunderWidth =
      options_.GetOptionsDict().Get<float>(kDeblunderQBlunderWidth);

  SyzygyTablebase tablebase;
  if (!tablebase.init(
          options_.GetOptionsDict().Get<std::string>(kSyzygyTablebaseId)) ||
      tablebase.max_cardinality() < 3) {
    std::cerr << "FAILED TO LOAD SYZYGY" << std::endl;
    return;
  }
  auto dtmPaths =
      options_.GetOptionsDict().Get<std::string>(kGaviotaTablebaseId);
  if (dtmPaths.size() != 0) {
    std::stringstream path_string_stream(dtmPaths);
    std::string path;
    auto paths = tbpaths_init();
    while (std::getline(path_string_stream, path, SEP_CHAR)) {
      paths = tbpaths_add(paths, path.c_str());
    }
    tb_init(0, tb_CP4, paths);
    tbcache_init(64 * 1024 * 1024, 64);
    if (tb_availability() != 63) {
      std::cerr << "UNEXPECTED gaviota availability" << std::endl;
      return;
    } else {
      std::cerr << "Found Gaviota TBs" << std::endl;
    }
    gaviotaEnabled = true;
  }
  auto policySubsDir =
      options_.GetOptionsDict().Get<std::string>(kPolicySubsDirId);
  if (policySubsDir.size() != 0) {
    auto policySubFiles = GetFileList(policySubsDir);
    for (long unsigned int i = 0; i < policySubFiles.size(); i++) {
      policySubFiles[i] = policySubsDir + "/" + policySubFiles[i];
    }
    BuildSubs(policySubFiles);
  }

  auto inputDir = options_.GetOptionsDict().Get<std::string>(kInputDirId);
  if (inputDir.size() == 0) {
    std::cerr << "Must provide an input dir." << std::endl;
    return;
  }
  auto files = GetFileList(inputDir);
  if (files.size() == 0) {
    std::cerr << "No files to process" << std::endl;
    return;
  }
  for (long unsigned int i = 0; i < files.size(); i++) {
    files[i] = inputDir + "/" + files[i];
  }
  float dtz_boost = options_.GetOptionsDict().Get<float>(kMinDTZBoostId);
  long unsigned int threads = options_.GetOptionsDict().Get<int>(kThreadsId);
  ProcessFileFlags flags;
  flags.delete_files = options_.GetOptionsDict().Get<bool>(kDeleteFilesId);
  flags.nnue_best_score = options_.GetOptionsDict().Get<bool>(kNnueBestScoreId);
  flags.nnue_best_move = options_.GetOptionsDict().Get<bool>(kNnueBestMoveId);
  if (threads > 1) {
    std::vector<std::thread> threads_;
    int offset = 0;
    while (threads_.size() < threads) {
      int offset_val = offset;
      offset++;
      threads_.emplace_back([this, offset_val, files, &tablebase, threads,
                             dtz_boost, flags]() {
        ProcessFiles(
            files, &tablebase,
            options_.GetOptionsDict().Get<std::string>(kOutputDirId),
            options_.GetOptionsDict().Get<float>(kDeltaQId),
            options_.GetOptionsDict().Get<int>(kNumberOfPiecesId),
            dtz_boost, options_.GetOptionsDict().Get<int>(kNewInputFormatId),
            offset_val, threads,
            options_.GetOptionsDict().Get<std::string>(kNnuePlainFileId),
            flags);
      });
    }
    for (long unsigned int i = 0; i < threads_.size(); i++) {
      threads_[i].join();
    }

  } else {
    ProcessFiles(files, &tablebase,
                 options_.GetOptionsDict().Get<std::string>(kOutputDirId),
                 options_.GetOptionsDict().Get<float>(kDeltaQId),
                 options_.GetOptionsDict().Get<int>(kNumberOfPiecesId),
                 dtz_boost,
                 options_.GetOptionsDict().Get<int>(kNewInputFormatId), 0, 1,
                 options_.GetOptionsDict().Get<std::string>(kNnuePlainFileId),
                 flags);
  }
  // std::cout << "Games processed: " << games << std::endl;
  // std::cout << "Positions processed: " << positions << std::endl;
  // std::cout << "Rescores performed: " << rescored << std::endl;
  // std::cout << "Cumulative outcome change: " << delta << std::endl;
  // std::cout << "Secondary rescores performed: " << rescored2 << std::endl;
  // std::cout << "Secondary rescores performed used dtz: " << rescored3
  //           << std::endl;
  // std::cout << "Blunders picked up by deblunder threshold: " << blunders
  //           << std::endl;
  // std::cout << "Number of policy values boosted by dtz or dtm " << policy_bump
  //           << std::endl;
  // std::cout << "Number of policy values boosted by dtm " << policy_dtm_bump
  //           << std::endl;
  // std::cout << "Orig policy_sum dist of boost candidate:";
  // std::cout << std::endl;
  // int event_sum = 0;
  // for (int i = 0; i < 11; i++) event_sum += policy_bump_total_hist[i];
  // for (int i = 0; i < 11; i++) {
  //   std::cout << " " << std::setprecision(4)
  //             << ((float)policy_nobump_total_hist[i] / (float)event_sum);
  // }
  // std::cout << std::endl;
  // std::cout << "Boosted policy_sum dist of boost candidate:";
  // std::cout << std::endl;
  // for (int i = 0; i < 11; i++) {
  //   std::cout << " " << std::setprecision(4)
  //             << ((float)policy_bump_total_hist[i] / (float)event_sum);
  // }
  // std::cout << std::endl;
  // std::cout << "Original L: " << orig_counts[0] << " D: " << orig_counts[1]
  //           << " W: " << orig_counts[2] << std::endl;
  // std::cout << "After L: " << fixed_counts[0] << " D: " << fixed_counts[1]
  //           << " W: " << fixed_counts[2] << std::endl;
  // std::cout << "Gaviota DTM move_count rescores: " << gaviota_dtm_rescores
  //           << std::endl;
}

SelfPlayLoop::SelfPlayLoop() {}

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::RunLoop() {
  SelfPlayTournament::PopulateOptions(&options_);

  options_.Add<BoolOption>(kInteractiveId) = false;
  options_.Add<StringOption>(kLogFileId);

  if (!options_.ProcessAllFlags()) return;

  Logging::Get().SetFilename(
      options_.GetOptionsDict().Get<std::string>(kLogFileId));

  if (options_.GetOptionsDict().Get<bool>(kInteractiveId)) {
    UciLoop::RunLoop();
  } else {
    // Send id before starting tournament to allow wrapping client to know
    // who we are.
    // SendId();
    SelfPlayTournament tournament(
        options_.GetOptionsDict(),
        std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
        std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
    tournament.RunBlocking();
  }
}

void SelfPlayLoop::CmdUci() {
  // SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void SelfPlayLoop::CmdStart() {
  if (tournament_) return;
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
}

void SelfPlayLoop::CmdStop() {
  tournament_->Stop();
  tournament_->Wait();
}

void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::vector<std::string> responses;
  // Send separate resign report before gameready as client gameready parsing
  // will easily get confused by adding new parameters as both training file
  // and move list potentially contain spaces.
  if (info.min_false_positive_threshold) {
    std::string resign_res = "resign_report";
    resign_res +=
        " fp_threshold " + std::to_string(*info.min_false_positive_threshold);
    responses.push_back(resign_res);
  }
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  res += " play_start_ply " + std::to_string(info.play_start_ply);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameResult::UNDECIDED) {
    res += std::string(" result ") +
           ((info.game_result == GameResult::DRAW)
                ? "draw"
                : (info.game_result == GameResult::WHITE_WON) ? "whitewon"
                                                              : "blackwon");
  }
  if (!info.moves.empty()) {
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.as_string();
  }
  if (!info.initial_fen.empty() &&
      info.initial_fen != ChessBoard::kStartposFen) {
    res += " from_fen " + info.initial_fen;
  }
  responses.push_back(res);
  SendResponses(responses);
}

void SelfPlayLoop::CmdSetOption(const std::string& name,
                                const std::string& value,
                                const std::string& context) {
  options_.SetUciOption(name, value, context);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  const int winp1 = info.results[0][0] + info.results[0][1];
  const int losep1 = info.results[2][0] + info.results[2][1];
  const int draws = info.results[1][0] + info.results[1][1];

  // Initialize variables.
  float percentage = -1;
  std::optional<float> elo;
  std::optional<float> los;

  // Only caculate percentage if any games at all (avoid divide by 0).
  if ((winp1 + losep1 + draws) > 0) {
    percentage =
        (static_cast<float>(draws) / 2 + winp1) / (winp1 + losep1 + draws);
  }
  // Calculate elo and los if percentage strictly between 0 and 1 (avoids divide
  // by 0 or overflow).
  if ((percentage < 1) && (percentage > 0))
    elo = -400 * log(1 / percentage - 1) / log(10);
  if ((winp1 + losep1) > 0) {
    los = .5f +
          .5f * std::erf((winp1 - losep1) / std::sqrt(2.0 * (winp1 + losep1)));
  }
  std::ostringstream oss;
  oss << "tournamentstatus";
  if (info.finished) oss << " final";
  oss << " P1: +" << winp1 << " -" << losep1 << " =" << draws;

  if (percentage > 0) {
    oss << " Win: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (percentage * 100.0f) << "%";
  }
  if (elo) {
    oss << " Elo: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*elo);
  }
  if (los) {
    oss << " LOS: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*los * 100.0f) << "%";
  }

  oss << " P1-W: +" << info.results[0][0] << " -" << info.results[2][0] << " ="
      << info.results[1][0];
  oss << " P1-B: +" << info.results[0][1] << " -" << info.results[2][1] << " ="
      << info.results[1][1];
  oss << " npm " + std::to_string(static_cast<double>(info.nodes_total_) /
                                  info.move_count_);
  oss << " nodes " + std::to_string(info.nodes_total_);
  oss << " moves " + std::to_string(info.move_count_);
  SendResponse(oss.str());
}

}  // namespace lczero
