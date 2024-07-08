#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H

#include "marco/Codegen/Lowering/Distance/WordDistanceCalculator.h"

/// @file SentenceDistanceCalculator.h
/// @brief Header file for the SentenceDistanceCalculator class.
/// 
/// This is a custom implementation of a semantic distance calculator
/// between two variable names. The idea is to calculate the distance
/// between two variable names by interpreting them as short sentences.
/// The distance is calculated by following the paper "Sentence similarity
/// based on semantic nets and corpus statistics" by Li, McLean, Bandar,
/// O'Shea, Crockett.
/// 
/// Note that the paper uses a special process to give semantic weight to
/// the orderings of words in a sentence. This is not implemented in this
/// project, as the application (specific to finding similarity between
/// variable names in scripting) does not require it. Furthermore, implementing
/// this would actually be detrimental to the application, as the order of
/// words in a variable name is marginally important, and would only serve
/// to reduce all similarities, and make it harder to find the correct variable name.

namespace marco::codegen::lowering
{
    /// @class marco::codegen::lowering::SentenceDistanceCalculator
    /// @brief A class that calculates the distance between two sentences.
    /// 
    /// In our context, a sentence is a variable name, as variable names
    /// can be seen as short sentences that describe the variable's purpose.
    /// These sentences may not contain verbs or other parts of speech, but
    /// the relationship between the words in the sentence can still be
    /// analyzed. The task of comparing two sentences is done by this class,
    /// following the paper "Sentence similarity based on semantic nets and
    /// corpus statistics" by Li, McLean, Bandar, O'Shea, Crockett.
    class SentenceDistanceCalculator {
    private:
        /// @brief The WordDistanceCalculator instance used to calculate
        /// the distance between two words.
        WordDistanceCalculator wordDistanceCalculator;

        /// @brief Lowercase a string.
        static void lowerCase(std::string& str);

        /// @brief Split a camel case string into words.
        static std::vector<std::string> camelCaseSplit(llvm::StringRef str);
        
        /// @brief Split an underscore-separated string into words.
        static std::vector<std::string> underscoreSplit(llvm::StringRef str);
        
        /// @brief Split a string into words.
        /// 
        /// Note that this function infers the splitting method based on the
        /// presence of underscores in the string.
        static std::vector<std::string> split(llvm::StringRef str);

        /// @brief Get the joint word set of two sentences.
        /// 
        /// A joint word set is the set of all unique words in two sentences.
        /// The first sentence is copied as is, excluding duplicate words.
        /// The second sentence is then added to the joint word set, excluding
        /// words that are already in the first sentence and excluding duplicate
        /// words.
        static std::vector<std::string> getJointWordSet(llvm::ArrayRef<std::string> sentence1,
                                                        llvm::ArrayRef<std::string> sentence2);

        /// @brief Get the lexical cell of a string with respect to a sentence.
        /// 
        /// A lexical cell is a measure of how much a string is related to a
        /// sentence. This measure is used to calculate the similarity between
        /// word vectors. See getWordVecSimilarity() and the aforementioned paper
        /// for more information.
        /// 
        /// To gather this score, the function calculates the semantic distance
        /// between the string and each word in the sentence, and returns the
        /// maximum distance.
        float getLexicalCell(llvm::StringRef str, llvm::ArrayRef<std::string> sentence);

        /// @brief Get the similarity between two word vectors.
        /// 
        /// This function calculates the cosine similarity between two word
        /// vectors. The word vectors are generated by the getLexicalCell()
        /// function, which calculates the similarity between a string and
        /// a sentence.
        float getWordVecSimilarity(llvm::ArrayRef<std::string> sentence1,
                                   llvm::ArrayRef<std::string> sentence2);

    public:
        SentenceDistanceCalculator();

        /// @brief Get the similarity between two sentences.
        /// 
        /// This is a public method that calculates the similarity between
        /// two sentences. The sentences are split into words, and thed
        /// similarity is calculated by comparing the word vectors.
        /// 
        /// This is the final product of the sentence distance
        /// calculation, and is used to determine the similarity between two
        /// variable names within marco, if the classical distance calculation
        /// reports unsatisfactory results.
        float getSimilarity(llvm::StringRef sentence1,
                            llvm::StringRef sentence2);
    };
}

#endif //MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H