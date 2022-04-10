import numpy as np


def clean_path(path):
    """utility function that performs basic text cleaning on path"""

    # No need to modify
    path = str(path).replace("'", "")
    path = path.replace(",", "")
    path = path.replace(" ", "")
    path = path.replace("[", "")
    path = path.replace("]", "")

    return path


class GreedySearchDecoder(object):
    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """
        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """
        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        print("y_probs: ", y_probs.T)
        last_sym = None
        for i in range(len(y_probs[0])):
            max_prob = np.max(y_probs[:, i, 0], axis=0)
            max_index = np.argmax(y_probs[:, i, 0], axis=0)
            # max_prob = 0
            # for j in range(len(y_probs)):
            #     if y_probs[j][i][0] > max_prob:
            #         max_prob = y_probs[j][i][0]
            #         max_index = j
            path_prob *= max_prob
            path_symbol = self.symbol_set[max_index - 1]
            if max_index == blank:
                decoded_path.append(" ")
            else:
                if last_sym != path_symbol:
                    decoded_path.append(path_symbol)
                    last_sym = path_symbol
        decoded_path = clean_path(decoded_path)

        # for i in range(y_probs.shape[1]):
        #     max_prob = np.max(y_probs[:, i, 0], axis=0)
        #     max_index = np.argmax(y_probs[:, i, 0], axis=0)
        #     print("\nMax", max_prob, max_index)
        #     path_prob *= max_prob
        #     if max_index == 0:
        #         decoded_path.append(",")
        #     else:
        #         decoded_path.append(self.symbol_set[max_index - 1])
        # decoded_path = clean_path(decoded_path)

        return (decoded_path, path_prob)


class BeamSearchDecoder(object):
    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """
        self.blank = 0
        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def initialize_paths(self, y0_probs):
        blank_path_score = dict()
        path_score = dict()
        init_paths_with_final_blank = list()
        init_paths_with_final_symbol = list()

        # Blank
        path = ""
        blank_path_score[path] = y0_probs[self.blank]
        init_paths_with_final_blank.append(path)

        # Other Symbols
        for i, sym in enumerate(self.symbol_set):
            path_score[sym] = y0_probs[i + 1]
            init_paths_with_final_symbol.append(sym)

        return init_paths_with_final_blank, init_paths_with_final_symbol, blank_path_score, path_score

    def extend_with_blank(self, paths_with_terminal_blank, paths_with_terminal_symbol, blank_path_score, path_score, y_probs):
        updated_paths_with_terminal_blank = list()
        updated_blank_path_scores = dict()

        for path in paths_with_terminal_blank:
            updated_paths_with_terminal_blank.append(path)
            updated_blank_path_scores[path] = blank_path_score[path] * y_probs[self.blank]

        for path in paths_with_terminal_symbol:
            if path in updated_paths_with_terminal_blank:
                updated_blank_path_scores[path] += path_score[path] * y_probs[self.blank]
            else:
                updated_paths_with_terminal_blank.append(path)
                updated_blank_path_scores[path] = path_score[path] * y_probs[self.blank]

        return updated_paths_with_terminal_blank, updated_blank_path_scores

    def extend_with_symbol(self, paths_with_terminal_blank, paths_with_terminal_symbol, blank_path_score, path_score, y_probs):
        updated_paths_with_terminal_symbol = list()
        updated_path_scores = dict()

        for path in paths_with_terminal_blank:
            for i, sym in enumerate(self.symbol_set):
                new_path = path + sym  # concatenate
                updated_paths_with_terminal_symbol.append(new_path)
                updated_path_scores[new_path] = blank_path_score[path] * y_probs[i + 1]

        for path in paths_with_terminal_symbol:
            for i, sym in enumerate(self.symbol_set):
                newpath = path if sym == path[-1] else path + sym
                if newpath in updated_paths_with_terminal_symbol:
                    updated_path_scores[newpath] += path_score[path] * y_probs[i + 1]
                else:
                    updated_paths_with_terminal_symbol.append(newpath)
                    updated_path_scores[newpath] = path_score[path] * y_probs[i + 1]

        return updated_paths_with_terminal_symbol, updated_path_scores

    def prune(
        self,
        paths_with_terminal_blank,
        paths_with_terminal_symbol,
        blank_path_score,
        path_score,
    ):
        pruned_blank_path_scores = dict()
        pruned_path_scores = dict()
        pruned_paths_with_terminal_blank = list()
        pruned_paths_with_terminal_symbol = list()
        score_list = []

        for path in paths_with_terminal_blank:
            score_list.append(blank_path_score[path])

        for path in paths_with_terminal_symbol:
            score_list.append(path_score[path])

        score_list.sort(reverse=True)
        min_score = score_list[self.beam_width] if self.beam_width < len(score_list) else score_list[-1]

        for path in paths_with_terminal_blank:
            if blank_path_score[path] > min_score:
                pruned_paths_with_terminal_blank.append(path)
                pruned_blank_path_scores[path] = blank_path_score[path]

        for path in paths_with_terminal_symbol:
            if path_score[path] > min_score:
                pruned_paths_with_terminal_symbol.append(path)
                pruned_path_scores[path] = path_score[path]

        return pruned_paths_with_terminal_blank, pruned_paths_with_terminal_symbol, pruned_blank_path_scores, pruned_path_scores

    def merge_identical_paths(
        self,
        paths_with_terminal_blank,
        paths_with_terminal_symbol,
        blank_path_score,
        path_score,
    ):
        merged_paths = paths_with_terminal_symbol
        final_path_scores = path_score

        for path in paths_with_terminal_blank:
            if path in merged_paths:
                final_path_scores[path] += blank_path_score[path]
            else:
                merged_paths.append(path)
                final_path_scores[path] = blank_path_score[path]

        return merged_paths, final_path_scores

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        decoded_path = []
        sequences = [[list(), 1.0]]
        ordered = None

        best_path, merged_path_scores = None, None

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return

        new_paths_with_terminal_blank, new_paths_with_terminal_symbol, new_blank_path_score, new_path_score = self.initialize_paths(
            y_probs[:, 0]
        )
        for t in range(1, y_probs.shape[1]):
            terminal_blank_paths, terminal_symbol_paths, blank_path_score, path_score = self.prune(
                new_paths_with_terminal_blank, new_paths_with_terminal_symbol, new_blank_path_score, new_path_score
            )
            new_paths_with_terminal_blank, new_blank_path_score = self.extend_with_blank(
                terminal_blank_paths, terminal_symbol_paths, blank_path_score, path_score, y_probs[:, t]
            )
            new_paths_with_terminal_symbol, new_path_score = self.extend_with_symbol(
                terminal_blank_paths, terminal_symbol_paths, blank_path_score, path_score, y_probs[:, t]
            )

        _, merged_path_scores = self.merge_identical_paths(
            new_paths_with_terminal_blank, new_paths_with_terminal_symbol, new_blank_path_score, new_path_score
        )
        best_path = max(merged_path_scores, key=merged_path_scores.get)  # Find the path with the best score
        return (best_path, merged_path_scores)
