# ======================================================== #
# Main Class
# 
# Montage Model
#
# Bootstraps off of the transformer row rankings
# Combines inference patterns to build explanations
# ======================================================== #

import utils

import tablestore as ts
from patterns import loadPatterns
from qc_predictions import loadQCMap
from single_hop import pickHighProbabilityRow

from io import StringIO
from json import load as json_load
from json import dumps as json_dumps
from time import time
from collections import Counter

import pandas as pd
import os

def _computeNumOvl(A, B):
    cnt = 0
    for a in A:
        if a in B: cnt += 1
    return cnt

def _loadRowRankings(data_dir, model, use_large_rowRankingCache):
    print(f"Loading row rankings into memory...")

    rowRankingsMap = {}

    # Row rankings source directory
    rowRankings_dir = f"{data_dir}/rowRankings/{model}/"

    # Row rankings cache directory
    lg_sm_str = "lg" if use_large_rowRankingCache else "sm"
    rowRankings_cache_addr = f"{data_dir}/rowRankings/rowRankings_{model}_cache_{lg_sm_str}.json"

    refresh_cache = utils.isCacheOld(rowRankings_dir, rowRankings_cache_addr)
    if refresh_cache:
        print(f"\tWarning! Cache not up to date, reparsing original row rankings files from \"{rowRankings_dir}\"")
        
        # Only keep the top relevant facts
        if not use_large_rowRankingCache:
            N = 200 if model == 'tfidf' else 100
            N += 1 # header

        # N = -2 # header

        # Popuilate this map with info from every question file
        # Hash by QID
        rowRankingsMap = {}
        
        for split in ["Train", "Test", "Dev"]:
            rankingsDir = f"{rowRankings_dir}/rankings.{split}/"
            questionFiles = os.listdir(rankingsDir)

            # User info
            numQs = len(questionFiles)
            ticker = utils.Ticker(numQs, tick_rate=121, message=f"Loading {numQs} {split} questions...")    
            
            for rowRankingFileName in questionFiles:

                # info for user
                ticker.tick()

                with open(rankingsDir+rowRankingFileName, mode="r") as fp:
                    # question ID is in the file name rowRanking_{qid}.tsv
                    qid = rowRankingFileName[11:-4]
                    lines = fp.read().split("\n")

                    # Grab the header and the lines
                    header = lines[0].split("\t")
                    lines = lines[1:-1] if use_large_rowRankingCache else lines[1:N+1]
                    if use_large_rowRankingCache: N = len(lines)

                    # Make a JSON for the tsv
                    rankingJSON = {head:[""]*N for head in header}
                    rankingJSON['length'] = N
                    for i, line in enumerate(lines):
                        cols = line.split("\t")

                        for j, col in enumerate(cols):
                            rankingJSON[header[j]][i] = col

                    # Save the JSON in the full export obj
                    rowRankingsMap[qid] = rankingJSON

            ticker.end()

        # Convert stuff to floats
        print("Converting")
        for qid in rowRankingsMap.keys():
            for i, isGoldRow in enumerate(rowRankingsMap[qid]['isGoldRow']):
                rowRankingsMap[qid]['isGoldRow'][i] = int(isGoldRow)
            for i, score in enumerate(rowRankingsMap[qid]['score_1']):
                rowRankingsMap[qid]['score_1'][i] = float(score)

        print(f"Caching row rankings to \"{rowRankings_cache_addr}\".")
        with open(rowRankings_cache_addr, mode="w") as fp:
            fp.write(json_dumps(rowRankingsMap))
    else:
        print(f"\tReading cached row rankings from \"{rowRankings_cache_addr}\".")
        with open(rowRankings_cache_addr, mode="r") as fp:
            rowRankingsMap = json_load(fp)

            # Convert stuff to floats
            for qid in rowRankingsMap.keys():
                for i, isGoldRow in enumerate(rowRankingsMap[qid]['isGoldRow']):
                    rowRankingsMap[qid]['isGoldRow'][i] = int(isGoldRow)
                for i, score in enumerate(rowRankingsMap[qid]['score_1']):
                    rowRankingsMap[qid]['score_1'][i] = float(score)

            print("Done.")

    return rowRankingsMap

class Montage:

    # ============= #
    # Class Methods #
    # ============= #

    def __init__(self,     
                rowRankings_dir,
                tablestore_dir,
                patterns_dir,
                qc_predictions_dir,

                cache_dir="cache/",
                qc_level=3,
                filter_perc=0.1,
                developer_mode = False,
                verbose=False
        ):

        # enusre directories end in "/"
        if not rowRankings_dir.endswith('/'): rowRankings_dir = rowRankings_dir + "/"
        if not tablestore_dir.endswith('/'): tablestore_dir = tablestore_dir + "/"
        if not patterns_dir.endswith('/'): patterns_dir = patterns_dir + "/"
        if not qc_predictions_dir.endswith('/'): qc_predictions_dir = qc_predictions_dir + "/"
        if not cache_dir.endswith('/'): cache_dir = cache_dir + "/"

        self.rowRankings_dir = rowRankings_dir
        self.tablestore_dir = tablestore_dir
        self.patterns_dir = patterns_dir
        self.qc_predictions_dir = qc_predictions_dir

        self.cache_dir = cache_dir

        self.qc_level = qc_level
        self.filter_perc = filter_perc

        self.developer_mode = developer_mode
        self.verbose = verbose

        # define a string that is associated wih this exact 
        bootstrap_model = rowRankings_dir.split("/")[-2]
        tablestoreName = tablestore_dir.split("/")[-2]
        patternsSetName = patterns_dir.split("/")[-2]
        qc_str = self.getQCLevelStr() # NOTE: this was abstracted before I wrote this line, so i'll leave it be
        self.configStr = f"rr_{bootstrap_model}.ts_{tablestoreName}.fp_{filter_perc}.p_{patternsSetName}.{qc_str}"

        # Define filter percentages
        #TODO: not do this manually
        if filter_perc == 5:
            self.filter_tol = 0.82
        elif filter_perc == 1:
            self.filter_tol = -0.80
        elif filter_perc == 0.1:
            self.filter_tol = -2.18
        elif filter_perc == 0:
            self.filter_tol = -5.33
        else:
            raise AssertionError

        print("==========================================================\n")

        # Load the tablestore
        tablestore = ts.loadTablestore(tablestoreDir=tablestore_dir, cache_dir=cache_dir)
        if tablestore:
            self.tablestore = tablestore

            # Load the patterns (comes with qc distrobutions for the classes)
            patterns, qc_dist_count, qc_dist_prop = loadPatterns(patterns_dir, qc_level, cache_dir=cache_dir)
            if patterns:
                self.patterns = patterns
                self.qc_dist_count = qc_dist_count
                self.qc_dist_prop = qc_dist_prop

                # Load Question Classifications 
                qcMap = loadQCMap(qc_predictions_dir, qc_level, questionsList=tablestore.getQuestionList(), cache_dir=cache_dir)
                if qcMap:
                    self.qcMap = qcMap
 
                    # Ensure that the enum features are generateed and on file
                    self.verifyEnumFeatures()

                    # Rank the enums for this config using the ML model
                    # if not self.verifyRankedEnums():
                    #     print("\t NOTE: ML model has not ranked the enumerations for necessary questions.\nRunning ranking task...\n")
                    #     from tf_ranking_libsvm import generate_predictions
                    #     from_split = ['Dev'] if developer_mode else ['Train', 'Test', 'Dev']
                    #     generate_predictions(
                    #         patterns_model=self, 
                    #         from_split=from_split
                    #     )

                    # Flag the model as loaded
                    print("Montage fully loaded.\n")
                    self.loaded = True

                else:
                    self.loaded = False
            else:
                self.loaded = False
        else:
            self.loaded = False

        print("==========================================================\n")

    # ======================== #
    # Getter Methods - Strings #
    # ======================== #

    def getFeaturesAddr(self, question: ts.Question):
        """Address of a question's enumeration features"""
        return self.getFeaturesDir(question.split) + self.getFeaturesFilename(question)

    def getFeaturesFilename(self, question: ts.Question):
        """Filename of a question's enumeration features"""
        return f"enumFeatures_{question.id}.tsv"

    def getFeaturesDir(self, split: str):
        """Directory where the features are stored and read"""
        directory = f"{self.cache_dir}/enumerationFeatures/enumFeatures.{self.getConfigStr()}.{split}/"
        if not os.path.exists(directory): os.makedirs(directory)
        return directory

    def getRankedEnumsAddr(self, question: ts.Question):
        """Address of a question's ranked enumerations"""
        return self.getRankedEnumsDir(question.split) + self.getRankedEnumsFilename(question)

    def getRankedEnumsFilename(self, question: ts.Question):
        """Filename of a question's ranked enumerations"""
        return f"enumRanking_{question.id}.tsv"

    def getRankedEnumsDir(self, split: str):
        """Directory where the ranked pattern enumerations are stored and read"""
        directory = f"{self.cache_dir}/enumRankings/enumRankings.{self.getConfigStr()}.{split}/"
        if not os.path.exists(directory): os.makedirs(directory)
        return directory
        
    def getQCLevelStr(self):
        """String for the question classification level"""
        return f"qc_L{self.qc_level}"

    def getConfigStr(self):
        """Get a string associated with the model config. Useful for naming output files."""
        return self.configStr

    # ========================== #
    # Getter Methods - Questions #
    # ========================== #

    def getQuestionList(self, from_split=[]):
        return self.tablestore.getQuestionList(from_split=from_split)

    def getQuestionMap(self, from_split=[]):
        """A map to each , hash by qid"""
        questionList = self.tablestore.getQuestionList(from_split=from_split)
        questionMap = {question.id: question for question in questionList}
        return questionMap

    def getQuestionsIDs(self):
        return self.tablestore.getQuestionsIDs()

    def getQuestion(self, qid: str):
        return self.tablestore.getQuestion(qid)

    def getQCs(self, qid: str, depth=5):
        qcs = self.qcMap[qid]
        return qcs[:5] if len(qcs) > 5 else qcs

    # ============================ #
    # Getter Methods - Ranked Rows #
    # ============================ #

    def getRankedRows(self, question: ts.Question):
        verbose = self.verbose
        split = question.split
        qid = question.id

        # Row rankings source directory
        rowRankings_addr = f"{self.rowRankings_dir}/rankings.{split}/rowRanking_{qid}.tsv"

        if verbose: print(f"Loading row ranking from \"{rowRankings_addr}\"...")
            
        # Parse the tsv into a JSON dataframe structure (i.e. map of lists)
        if os.path.exists(rowRankings_addr):
            with open(rowRankings_addr, mode="r") as fp:
                lines = fp.read().split("\n")

                # Grab the header and the lines
                header = lines[0].split("\t")
                lines = lines[1:]

                # Make a JSON for the tsv
                rankingJSON = {head:[] for head in header}
                for line in lines:
                    if line != "":
                        cols = line.split("\t")

                        for col, head in zip(cols, header):

                            # Convert the numerical columns into floats
                            if head == "isGoldRow" or head == "score_0" or head == "score_1":
                                rankingJSON[head].append(float(col))
                            else:
                                rankingJSON[head].append(col)

                return rankingJSON
            
            print(f"\tERROR! Read file but did not return data...")
            return None
        else:
            print(f"\tERROR! Cannot find ranked rows for model \"{model}\", question \"{qid}\".\nLooking in \"{rowRankings_addr}\".")
            return None

    # ========================= #
    # Getter Methods - Patterns #
    # ========================= #

    def getPatternDistrobutions(self, qid: str):

        # Question classes [[qc_, w_0],...]
        qcs = self.getQCs(qid, depth=5)

        # Get the pattern distrobution info for each class this question is clf as
        qc_dist_count = self.qc_dist_count
        qc_dist_prop = self.qc_dist_prop

        q_pattern_dist_counts = []
        for qc in qcs:
            key = qc[0]
            weight = qc[1]
            if key in qc_dist_count.keys():
                q_pattern_dist_counts.append((qc_dist_count[key], weight))
            else:
                print(f"\tWARNING! No count distrobution for qc \"{key}\"")

        q_pattern_dist_props = []
        for qc in qcs:
            key = qc[0]
            weight = qc[1]
            if key in qc_dist_prop.keys():
                q_pattern_dist_props.append((qc_dist_prop[key], weight))
            else:
                print(f"\tWARNING! No proportion distrobution for qc \"{key}\"")

        # =================================================================== #
        # Combine all the distrobutions for each class, into one distrobution #
        # Store the qc class weights                                          #
        # just sum everything                                                 #
        # =================================================================== #
        
        # counts
        q_pattern_dist_count = {}
        len_q_pattern_dist_counts = len(q_pattern_dist_counts)
        if len_q_pattern_dist_counts > 0:
            q_pattern_dist_counts_0 = q_pattern_dist_counts[0]

            # Pull out the distrobution and the weight for the qc that this distrobution is for

                                                        # Deep copy
            q_pattern_dist_count = {key: value for key, value in q_pattern_dist_counts_0[0].items()}
            qc_weight = q_pattern_dist_counts_0[1]

            # Convert the floats into a list (one probability for each qc)
            for key in q_pattern_dist_count.keys():
                q_pattern_dist_count[key] = [q_pattern_dist_count[key]]

            # Store the qc weight 
            q_pattern_dist_count["__qcWeights__"] = [qc_weight]

            # If there are more classes for this questions add them to the lists
            if len_q_pattern_dist_counts > 1:
                q_pattern_dist_counts = q_pattern_dist_counts[1:]

                for q_pattern_dist_count_i, qc_weight_i in q_pattern_dist_counts:
                    q_pattern_dist_count["__qcWeights__"].append(qc_weight_i)

                    for key in q_pattern_dist_count_i.keys():
                        q_pattern_dist_count[key].append(q_pattern_dist_count_i[key])
        else:
            q_pattern_dist_count = None

        # proportions
        q_pattern_dist_prop = {}
        len_q_pattern_dist_props = len(q_pattern_dist_props)
        if len_q_pattern_dist_props > 0:
            q_pattern_dist_props_0 = q_pattern_dist_props[0]

            # Pull out the distrobution and the weight for the qc that this distrobution is for

                                                        # Deep copy
            q_pattern_dist_prop = {key: value for key, value in q_pattern_dist_props_0[0].items()}
            qc_weight = q_pattern_dist_props_0[1]

            # Convert the floats into a list (one probability for each qc)
            for key in q_pattern_dist_prop.keys():
                q_pattern_dist_prop[key] = [q_pattern_dist_prop[key]]

            # Store the qc weight 
            q_pattern_dist_prop["__qcWeights__"] = [qc_weight]

            # If there are more classes for this questions add them to the lists
            if len_q_pattern_dist_props > 1:
                q_pattern_dist_props = q_pattern_dist_props[1:]

                for q_pattern_dist_prop_i, qc_weight_i in q_pattern_dist_props:
                    q_pattern_dist_prop["__qcWeights__"].append(qc_weight_i)

                    for key in q_pattern_dist_prop_i.keys():
                        q_pattern_dist_prop[key].append(q_pattern_dist_prop_i[key])
        else:
            q_pattern_dist_prop = None

        return q_pattern_dist_count, q_pattern_dist_prop

    def getPatternEnumerations(self, patternName: str):
        """
            Retrive a list of enumerations for a pattern by pattern name

            input: 
                patternName - Name of the pattern
            return:
                A list of enumerations for the pattern
        """
        return self.patterns[patternName]

    def getRankedEnumerationFeatures(self, question: ts.Question):
        """Get the ranked dataframe for the fetrues for the given question"""
        rankedEnumerationsFeaturesAddr = self.getRankedEnumsAddr(question)
        if os.path.exists(rankedEnumerationsFeaturesAddr):
            return pd.read_csv(rankedEnumerationsFeaturesAddr, sep='\t')
        else:
            print(f"\tWARNING! Missing ranked enumerations for question \"{question.id}\".")
            return pd.DataFrame()
        
    def getEnumerationFeatures(self, question: ts.Question):
        """
            Gets the enumeration features for the given question.
            If features are on filem, then it will load the featrures from there,
            if there are not then it will compute them live and save the computed features to a local cache.
            
            input: 
                question - Map that contains the split('category'), questionID('qid')
            return: 
                pandas dataframe of features for each enumeration
        """

        qid = question.id
        split = question.split

        # Address of the enumerations features
        enumerationFeaturesAddr = self.getFeaturesAddr(question)

        # If cache is on file use that, otherwise compute the features
        if os.path.exists(enumerationFeaturesAddr):
            # print(f"Loading pre-cached features for question \"{qid}\" from \"{enumerationFeaturesAddr}\".")
            features = pd.read_csv(enumerationFeaturesAddr, sep='\t')
            features = features[features.enumIdx.notnull()]
            return features
        else:
            print(f"Computing features for question \"{qid}\".")

            # Compute the enumeration features for this question
            features = self.determineEnumerationFeatures(question)

            # Write to file
            print(f"\tCaching features for question \"{qid}\" in \"{enumerationFeaturesAddr}\".")
            enumerationsFeaturesDir = self.getFeaturesDir(split)
            if not os.path.exists(enumerationsFeaturesDir): os.makedirs(enumerationsFeaturesDir)
            features.to_csv(enumerationFeaturesAddr, sep="\t", index=False)

            return features

    # ===================== #
    # Getter Methods - Misc #
    # ===================== #

    def getRow(self, uuid):
        return self.tablestore.getRow(uuid)

    def getTableNameForUUID(self, uuid):
        """Get the tablename for a given UUID"""
        row = self.tablestore.getRow(uuid)
        if row:
            return row.getTableName()
        else:
            return None

    def getFeatureNames(self):
        """Returns a list of the feature names"""
        return [
            "pattern_dist_counts_sum",
            "pattern_dist_prop_sum",
            "pattern_dist_counts_max",
            "pattern_dist_prop_max",
            "pattern_dist_counts_min",
            "pattern_dist_prop_min",
            "pattern_dist_counts_avg",
            "pattern_dist_prop_avg",
            "question_lemma_ovl",
            "answer_lemma_ovl",
            "ranked_scores",
            "ranked_index_radial",
            "ranked_scores_norm",
            "ranked_index_radial_norm"
        ]

    # ============= #
    # Funct Methods #
    # ============= #

    def verifyRankedEnums(self):
        """Verify that the ranked enums for each question are on file"""
        # Dev mode only needs DEV files
        from_split = ['Dev'] if self.developer_mode else ["Train", "Test", "Dev"]
        questions = self.getQuestionList(from_split=from_split)

        print(f'Checking to see if ML model has run on split(s): {", ".join(from_split)}.')

        # Verify that all files are on file
        allQuestionRankedEnumsOnFile = True
        missingQuestions = []
        for question in questions:
            if not os.path.exists( self.getRankedEnumsAddr(question) ):
                missingQuestions.append(question)
                allQuestionRankedEnumsOnFile = False
        
        # Print helpful info
        numMissingQuestions = len(missingQuestions)
        timeout = 5
        for question in missingQuestions:
            print(f"\tMissing ranked enumerations for {question.split} question \"{question.id}\".")
            timeout -= 1
            if timeout <= 0: break
        if numMissingQuestions > 5:
            print("\t...")
            print(f"\t{numMissingQuestions} questions missing.")

        return allQuestionRankedEnumsOnFile

    def verifyEnumFeatures(self):
        """Verify that the ranked enums for each question are on file"""
        # Dev mode only needs DEV files
        from_split = ['Dev'] if self.developer_mode else ["Train", "Test", "Dev"]
        questions = self.getQuestionList(from_split=from_split)

        print(f'Checking to see if ML model has run on split(s): {", ".join(from_split)}.')

        # Verify that all files are on file
        for question in questions:
            if not os.path.exists( self.getFeaturesAddr(question) ):
                self.getEnumerationFeatures(question)

    def determineEnumerationFeatures(self, question: ts.Question, num_features_cutoff=500, min_enumeration_count=5):
        """
            Compute the enumeration features for a given question.

            input: 
                question - Map that contains the split('category'), questionID('qid')
                num_features_cutoff - How many enumerations to keep when storing the features
                                      (NOTE: The exact number stored is specified by the minimum representation for each pattern.
                                      e.g. another +31 enumerations can be added to diversify the pattern represenatation)
                min_enumeration_count - Min representation for a pattern in enumeration shortlist. 
                                        (e.g. min of 5 enumerations from each pattern will be returned)
                                        (NOTE: Some patterns may not have at least 5 to return) 
            return: 
                pandas dataframe of features for each enumeration
        """
        t0 = time()

        filter_tol = self.filter_tol

        # Populate a list of features for the top enum for each pattern
        enumFeatures = []

        # question id
        qid = question.id

        # Get the patterns
        patterns = self.patterns
        patternNames = patterns.keys()

        # Get the ranked rows
        rankedRows = self.getRankedRows(question)
        rankedRowsUUIDs = rankedRows['uuid']
        rankedRowsScores = rankedRows['score_1']

        # Get the distrobutions for this question
        q_pattern_dist_counts, q_pattern_dist_props = self.getPatternDistrobutions(qid)

        # Get the quesiton lemmas
        question_lemmas = set(question.question_annotation['lemmas'])

        # get the correct answer lemmas
        answerkey = question.answerKey
        answer_lemmas = set(question.answer_annotation[answerkey]['lemmas'])

        # Get the golden question rows
        expl_gold_set = set(question.getExplanationUUIDs())

        # ============================================== #
        # Map uuids to weights in the bert row rankings  #
        # ============================================== #

        # Scores based on bert scoring
        rankedRowsScoresMap = dict(zip(rankedRowsUUIDs, rankedRowsScores))

        # Pure indices(1 indexed) radial scoring
        index_scores_radial = [1/(x+1) for x in range(len(rankedRowsUUIDs))]  
       
        # Maps
        rankedRowsIndexMap_radial = dict(zip(rankedRowsUUIDs, index_scores_radial))
       
        # ================================ #
        # Get weights for each enumeration #
        # ================================ #
        
        # Check to see if there are no patterns for this qc
        noPatternsForQC = True
        for patternName in patternNames:
            if 0 < sum(q_pattern_dist_counts[patternName]):
                noPatternsForQC = False
                break

        for patternName in patternNames:
            enumerations = patterns[patternName]

            # Get the top enum for this pattern
            enumFeature_top_score = -2**32
            enumFeature_top = None

            # ========================================== #
            # Get weights based on pattern distrobutions #
            # ========================================== #

            # Distrobution related features
            pattern_dist_counts = q_pattern_dist_counts[patternName]
            pattern_dist_props = q_pattern_dist_props[patternName]

            pattern_dist_counts_sum = sum(pattern_dist_counts)
            pattern_dist_prop_sum = sum(pattern_dist_props)

            # Dont use this pattern if there is 0 overlap
            # unless there are no patterns for this qc, then include all patterns
            if 0 < pattern_dist_counts_sum or noPatternsForQC:
                
                pattern_dist_counts_max = max(pattern_dist_counts)
                pattern_dist_prop_max = max(pattern_dist_props)

                pattern_dist_counts_min = min(pattern_dist_counts)
                pattern_dist_prop_min = min(pattern_dist_props)

                pattern_dist_counts_avg = pattern_dist_counts_sum / len(pattern_dist_counts)
                pattern_dist_prop_avg = pattern_dist_prop_sum / len(pattern_dist_props)

                for i_enum, enumeration in enumerate(enumerations):
                    uuids = enumeration['uuids']

                    # HACK: Skip enums that involve rows not in current copy of the tablestore
                    allUUIDGood = True
                    for uuid in uuids:
                        if not uuid.startswith("TEMPGEN") and uuid not in rankedRowsScoresMap.keys():
                            allUUIDGood = False
                    if not allUUIDGood: continue

                    # Filter out TEMPGEN and rows below the tolerance
                    uuids = set([uuid for uuid in uuids if not uuid.startswith("TEMPGEN") and filter_tol < rankedRowsScoresMap[uuid]])

                    if 0 == len(uuids): continue # Skip if we have filtered all the uuids

                    # Scores from the pre-rankings
                    ranked_scores = 0.0
                    ranked_index_radial = 0.0
                    for uuid in uuids:
                        ranked_scores += rankedRowsScoresMap[uuid]
                        ranked_index_radial += rankedRowsIndexMap_radial[uuid]

                    # If ranked score is higher than the max update the max enum
                    if enumFeature_top_score < ranked_scores:
                        enumFeature_top_score = ranked_scores

                        # Get the lemmas for this enumeration
                        lemmaSet = set()
                        for uuid in uuids:
                            row = self.getRow(uuid)
                            if row: lemmaSet = lemmaSet.union(row.getLemmaSet())

                        # Normalized versions of the pre-ranking scores
                        len_enum = len(uuids)
                        
                        ranked_scores_norm = ranked_scores / len_enum
                        ranked_index_radial_norm = ranked_index_radial / len_enum
                        
                        # Lemma overlap with the question
                        question_lemma_ovl = len(question_lemmas.intersection(lemmaSet))
                        answer_lemma_ovl = len(answer_lemmas.intersection(lemmaSet))

                        score_simple = len(uuids.intersection(expl_gold_set))
                        score_complex = self._score_complex(uuids.intersection(expl_gold_set))

                        # Store all features in a named dictionary
                        # add to the list
                        enumFeature_top = {
                            "patternName": patternName,
                            "enumIdx": i_enum,

                            "pattern_dist_counts_sum": pattern_dist_counts_sum,
                            "pattern_dist_prop_sum": pattern_dist_prop_sum,
                            "pattern_dist_counts_max": pattern_dist_counts_max,
                            "pattern_dist_prop_max": pattern_dist_prop_max,
                            "pattern_dist_counts_min": pattern_dist_counts_min,
                            "pattern_dist_prop_min": pattern_dist_prop_min,
                            "pattern_dist_counts_avg": pattern_dist_counts_avg,
                            "pattern_dist_prop_avg": pattern_dist_prop_avg,

                            "question_lemma_ovl": question_lemma_ovl,
                            "answer_lemma_ovl": answer_lemma_ovl,

                            "ranked_scores": ranked_scores,
                            "ranked_index_radial": ranked_index_radial,
                            
                            "ranked_scores_norm": ranked_scores_norm,
                            "ranked_index_radial_norm": ranked_index_radial_norm,

                            "score_simple": score_simple,
                            "score_complex": score_complex,
                        }
                        
                if enumFeature_top != None: enumFeatures.append(enumFeature_top)
        
        # =========================== #
        # Clean and convert to pandas #
        # =========================== #
        
        # Put them in a pandas dataframe
        features = pd.DataFrame(enumFeatures, columns=[
            "patternName",
            "enumIdx",
            "pattern_dist_counts_sum",
            "pattern_dist_prop_sum",
            "pattern_dist_counts_max",
            "pattern_dist_prop_max",
            "pattern_dist_counts_min",
            "pattern_dist_prop_min",
            "pattern_dist_counts_avg",
            "pattern_dist_prop_avg",
            "question_lemma_ovl",
            "answer_lemma_ovl",
            "ranked_scores",
            "ranked_index_radial",
            "ranked_scores_norm",
            "ranked_index_radial_norm",
            "score_simple",
            "score_complex",
        ])

        t1 = time()
        print(f"Time Elapsed: {t1-t0}s")

        # Return the features, but pre-sort them by ranked_scores
        features = features.sort_values(by='ranked_scores_norm', ascending=False)
        return features

    def _score_complex(self, uuids):
        """ 
            Score the uuids by table
            SYNONYMY: 0.1
            KINDOF    0.5
            <others>  1.0
        """
        score = 0.0
        for uuid in uuids:
            tableName = self.getTableNameForUUID(uuid)

            if tableName == 'SYNONOMY':
                score += 0.1
            elif tableName == 'KINDOF':
                score += 0.5
            elif tableName != None:
                score += 1.0
        return score

    def scoreFeatures(self, question:ts.Question, df_features: pd.DataFrame, scoringMethod: str):
        """
            Score each enumeration by its overlap with the gold explanation
            apply will do this for all rows essentially creating a new column of scores
            For each row get the pattern name and use that to look up the pattern enumerations,
            then get the enumIdx to index the enumerations,
            convert the enum uuids to a set and ovberlap with the gold rows
            return the length as the score
        """

        if scoringMethod == 'simple':
            return df_features['score_simple']
        elif scoringMethod == 'complex':
            return df_features['score_complex']
        else:
            print(f"\tERROR! Invalid scoring method: \"{scoringMethod}\".")
            return pd.DataFrame()

    def determineTopNEnumerations(self, question: ts.Question, top_N=25):

        # ============================================ #
        # Step 0) Precomputation / Determine features  #
        # ============================================ #

        # Internal parameters
        qid = question.id

        # Min required score for a row to be included in the explanation
        filter_tol = self.filter_tol

        # Patterns
        patterns = self.patterns

        # Dataframe of features for each enumeration relevant to the question
        features = self.getRankedEnumerationFeatures(question)

        # Generate a lookup map of the Scores based on ranked rows scoring
        rankedRows = self.getRankedRows(question)
        rankedRowsUUIDs = rankedRows['uuid']
        rankedRowsScores = rankedRows['score_1']
        rankedRowsScoresMap = dict(zip(rankedRowsUUIDs, rankedRowsScores))

        if 0 < features.shape[0]:

            # NOTE: This sorting should already be done
            features = features.sort_values(by='ranked_scores', ascending=False)
            # features = features.sort_values(by='ranked_weight', ascending=False)

            # Get the top 1 enumeration for each pattern
            enumerations_top = []
            for patternName in patterns.keys():
                
                # Get the enumerations for this pattern
                pattern_features = features[features['patternName'] == patternName]
                
                # Append the top 1 if it exists
                if pattern_features.shape[0] > 0:
                    enumeration = pattern_features.iloc[0]
                    enumerations_top.append(enumeration)
            
            # Convert to dataframe
            enumerations_top = pd.DataFrame(enumerations_top, columns=features.columns)

            # Sort by the ranking score
            enumerations_top = enumerations_top.sort_values(by='ranked_scores', ascending=False)
            # enumerations_top = enumerations_top.sort_values(by='ranked_weight', ascending=False)

            def getExplFromRow(row):
                patternName = row['patternName']
                enumIdx = int(row['enumIdx'])

                pattern = patterns[patternName][enumIdx]

                expl = [uuid for uuid in pattern['uuids'] if uuid in rankedRowsScoresMap.keys() and filter_tol < rankedRowsScoresMap[uuid]]

                expl += [uuid for uuid in rankedRowsUUIDs[:10] if uuid in rankedRowsScoresMap.keys() and filter_tol < rankedRowsScoresMap[uuid]]
                # expl += [uuid for uuid in rankedRowsUUIDs[:len(expl)] if uuid in rankedRowsScoresMap.keys() and filter_tol < rankedRowsScoresMap[uuid]]
                # expl += [uuid for uuid in rankedRowsUUIDs[:int(len(expl)/2)] if uuid in rankedRowsScoresMap.keys() and filter_tol < rankedRowsScoresMap[uuid]]
                    
                return set(expl)

            # Look up and store the UUIDs for each enumeration
            enumerations_top['uuids'] = enumerations_top.apply(getExplFromRow, axis=1)

            topNEnumerations = enumerations_top['uuids'][:top_N].values.tolist()

        else:
            # If no enums fall-back to ranked rows
            print(f"\tWARNING! No enumerations for question \"{qid}\", falling back to top 12 ranked rows.")
            topNEnumerations = [set() for _ in range(top_N)]
            topNEnumerations[0] = set(rankedRowsUUIDs[:12])

        # Return sorted enumerations
        return topNEnumerations

    def generateExpl(self, question: ts.Question):

        # ====================== #
        # Step 0) Precomputation #
        # ====================== #

        # Internal parameters
        qid = question.id

        # Min required score for a row to be included in the explanation
        filter_tol = self.filter_tol

        # Patterns
        patterns = self.patterns

        # Generate a lookup map of the Scores based on ranked rows scoring
        # NOTE: dont store past the filter tolerance
        rankedRows = self.getRankedRows(question)
        rankedRowsUUIDs = rankedRows['uuid']
        rankedRowsScores = []
        for (uuid, score) in zip(rankedRowsUUIDs, rankedRows['score_1']):
            if filter_tol <= score: rankedRowsScores.append((uuid, score))
            else: break
        rankedRowsScoresMap = dict(rankedRowsScores)

        # Dataframe of weighted enumerations relevant to the question
        # NOTE: If this is empty we have no info on this question, revert to baseline model
        df_enums = self.getEnumerationFeatures(question)
        if 0 < df_enums.shape[0]:

            # ============================== #
            # Step 1) Get the top 3 patterns #
            # ============================== #

            # Sort all enums by weight
            # ranked_scores - BERT score
            # ranked_weight - ML model score ( currently not better than BERT )
            df_enums = df_enums.sort_values(by='ranked_scores', ascending=False)
            # features = features.sort_values(by='ranked_weight', ascending=False)

            # Get the top 1 enumeration for each pattern
            # Build list then list -> dataframe
            enumerations_top = []
            for patternName in patterns.keys():
                
                # Get the enumerations for this pattern
                pattern_features = df_enums[df_enums['patternName'] == patternName]
                
                # Append the top 1 if it exists
                if pattern_features.shape[0] > 0:
                    enumeration = pattern_features.iloc[0]
                    enumerations_top.append(enumeration)
            
            # Convert to dataframe
            df_enums_top = pd.DataFrame(enumerations_top, columns=df_enums.columns)

            # Sort again since they just got sorted by pattern name
            df_enums_top = df_enums_top.sort_values(by='ranked_scores', ascending=False)
            # enumerations_top = enumerations_top.sort_values(by='ranked_weight', ascending=False)

            # Grab just the top 3
            df_enums_top = df_enums_top[:3]

            # Helper function for reading pattern enumerations from a pattern name + enum idx
            # This can be one-lined but I'm not a monster
            # lambda row: set([uuid for uuid in patterns[row['patternName']][int(row['enumIdx'])]['uuids'] if uuid in rankedRowsScoresMap.keys()])
            def getExplFromRow(row):
                patternName = row['patternName']
                enumIdx = int(row['enumIdx'])
                pattern = patterns[patternName][enumIdx]
                expl = [uuid for uuid in pattern['uuids'] if uuid in rankedRowsScoresMap.keys()]
                return set(expl)

            # Look up the UUIDs for each pattern
            # NOTE: apply returns a datframe, but we just want a list of uuid sets
            patternUUIDs = df_enums_top.apply(getExplFromRow, axis=1).values.tolist()

            # ================================ #
            # Step 2) Patterns to explanations #
            # ================================ #
            expls = []
            for n in range(3):
                # Join top n patterns
                expl = list(utils.joinSets(patternUUIDs[0:n]))

                # ======================== #
                # Step 3) Hybrid Component #
                # ======================== #
                # Add ranked rows 
                numRankedRows = 10
                # numRankedRows = len(expl)
                # numRankedRows = int(len(expl)/2)
                expl += [uuid for uuid in rankedRowsUUIDs[:numRankedRows] if uuid in rankedRowsScoresMap.keys()]

                # ============================ #
                # Step 4) Single-Hop Component #
                # ============================ #
                # Search through the
                # remaining top 20 ranked rows
                uuid = pickHighProbabilityRow(expl, rankedRows, question, self.tablestore)
                if uuid != None: expl.append(uuid)

                # Append the explanations list
                expls.append( expl )

            # Return explanations as a tuple
            return tuple(expls)

        else:
            # If no enums fall-back to ranked rows
            print(f"\tWARNING! No enumerations for question \"{qid}\", falling back to top 12 ranked rows.")
            return ( rankedRowsUUIDs[:12], [], [] )

if __name__ == "__main__":

    QC_DIR = "data/qc-predictions/"
    TABLESTORE_DIR = "data/tablestore/tablestore-2020-04-26/"
    PATTERNS_DIR = "data/patterns/auto/"
    RANKINGS_DIR = "data/rowRankings/bert/"
    CACHE_DIR      = "cache/"

    # ========== #
    # Load model #
    # ========== #
    model = Montage(
        rowRankings_dir = RANKINGS_DIR,
        tablestore_dir = TABLESTORE_DIR,
        patterns_dir = PATTERNS_DIR,
        qc_predictions_dir = QC_DIR,

        cache_dir=CACHE_DIR,
        qc_level=3,
        developer_mode = False,
        verbose=False
    )
