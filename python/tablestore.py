# ======================================================== #
# Support Class
# 
# Acts as an API to the tablestore
# ======================================================== #

import utils

import pandas as pd
import os
import json
import re
import spacy

from json import load as json_load
from json import dumps as json_dumps

class Question:

    def __init__(self, question_map: map):
        """
            Input: A map/df containing: category, QuestionID, explanation, question_annotation, answers_annotation, AnswerKey
        """

        self.split = question_map['category']
        self.id = question_map['QuestionID']
        self.explanation = [row.split('|') for row in question_map['explanation'].split(' ')]
        self.question_annotation = question_map["question_annotation"]
        self.answer_annotation = question_map["answers_annotation"]
        self.classes = question_map["topic"].split(", ")
        self.answerKey = question_map['AnswerKey']

    #========================#
    # Correct Answer Getters #
    #========================#
    def getCorrectAnswerAnnotation(self):
        """
            Retrieve the annnotated correct answer

            returns:
                The correct answer: lemms, words, pos tags
        """
        return self.answer_annotation[self.answerKey]

    def getCorrectAnswerWords(self):
        """
            Retrieve the annnotated correct answer

            returns:
                The correct answer: words
        """
        return self.answer_annotation[self.answerKey]['words']

    def getCorrectAnswerLemmas(self):
        """
            Retrieve the annnotated correct answer

            returns:
                The correct answer: lemmas
        """
        return self.answer_annotation[self.answerKey]['lemmas']

    def getCorrectAnswerPOS(self):
        """
            Retrieve the annnotated correct answer

            returns:
                The correct answer: pos tags
        """
        return self.answer_annotation[self.answerKey]['pos']

    def getCorrectAnswerText(self):
        """
            Cheap toString for the answer text.
        """
        return " ".join(self.answer_annotation[self.answerKey]['words'])

    #=======================#
    # Question Text Getters #
    #=======================#
    def getQuestionTextAnnotation(self):
        """
            Retrieve the annotated question text

            returns:
                Dict containing: lemmas, words, pos
        """
        return self.question_annotation

    def getQuestionTextWords(self):
        """
            Retrieve the annotated question text words

            returns:
                List of words
        """
        return self.question_annotation["words"]

    def getQuestionTextLemmas(self):
        """
            Retrieve the annotated question text lemmas

            returns:
                List of lemmas
        """
        return self.question_annotation["lemmas"]

    def getQuestionTextPOS(self):
        """
            Retrieve the annotated question text pos tags

            returns:
                List of pos tags
        """
        return self.question_annotation["pos"]

    def getQuestionText(self):
        """
            Get plain text from thew question
            NOTE: This is a cheap way of doing it rn
        """
        return " ".join(self.question_annotation['words'])

    #=====================#
    # Explanation Getters #
    #=====================#
    def getExplanationUUIDs(self, includeLexGlue=False):
        """
            Retrieve the uuids for the golden explanation

            returns:
                List of UUIDs
        """
        if includeLexGlue:
            return [row[0] for row in self.explanation]
        else:
            return [row[0] for row in self.explanation if row[1] != "LEXGLUE"]

    def getExplanationRoleMap(self):
        """Returns a map that stores the role for each row in the explanation"""
        return {row[0]: row[1] for row in self.explanation}

    #=======#
    # Misc. #
    #=======#
    def getClassList(self, level):
        """
            Retrieve the list of classes for this question

            input:
                level - how deep of a classification (1-6)

            returns:
                The correct answer: lemms, words, pos tags
        """
        if 1<=level and level <= 6: 
            return ["_".join(qc.split("_")[:level]) for qc in self.classes ]
        else:
            print(f"\tERROR! Invalid QC level {level}.")
            return []

    def getAnswersText(self):
        """Stringify all the answers, highlighting the correct one."""

        strOut = ""
        answerKey = self.answerKey
        for answer, annotation in self.answer_annotation.items():
            if answerKey == answer: strOut += f"[*] {' '.join(annotation['words'])}   "
            else:                   strOut += f"[] {' '.join(annotation['words'])}   "
        return strOut[:-3] # remove last 3 spaces

class TableRow:

    def __init__(self, row, tablestore):
        
        self.uuid = row["uuid"]
        self.tablename = row["tablename"]
        self.header = row["header"]
        self.headerSanitized = row["headerSanitized"]
        self.cells = row["cells"]
        self.cellWords = row["cellWords"]
        self.cellWordsLowerCase = row["cellWordsLowerCase"]
        self.cellLemmas = row["cellLemmas"]
        self.cellTags = row["cellTags"]

        self.length = len(self.header)

        self.tablestore = tablestore

        return

    def _isDataCol(self, head):
        return ("[SKIP]" not in head) and ("[FILL]" not in head)

    def _isSkipCol(self, head):
        return "[SKIP]" in head

    def _isFillCol(self, head):
        return "[FILL]" in head

    def getUUID(self):
        return self.uuid

    def getTableName(self):
        return self.tablename

    def getHeader(self):
        return self.header

    def getTextClean(self):
        words = self.getWordList(data_only=False, allow_noncontent_tags=True)
        return " ".join(words)

    def getLemmaSet(self, data_only=True, allow_noncontent_tags=False):
        lemmas = set()

        contentTags = self.tablestore.contentTags

        header = self.header
        cellTags = self.cellTags
        cellLemmas = self.cellLemmas
        for cell, tcell, head in zip(cellLemmas, cellTags, header):
            if not self._isSkipCol(head):
                if not (data_only and self._isFillCol(head)):
                    for alt, talt in zip(cell, tcell):
                        for lemma, tag in zip(alt, talt):
                            if tag[:2] in contentTags or allow_noncontent_tags:
                                lemmas.add(lemma)
        return lemmas

    def getWordSet(self, data_only=True, allow_noncontent_tags=False):
        words = set()

        contentTags = self.tablestore.contentTags

        header = self.header
        cellTags = self.cellTags
        cellWords = self.cellWords
        for cell, tcell, head in zip(cellWords, cellTags, header):
            if not self._isSkipCol(head):
                if not (data_only and self._isFillCol(head)):
                    for alt, talt in zip(cell, tcell):
                        for word, tag in zip(alt, talt):
                            if tag[:2] in contentTags or allow_noncontent_tags:
                                words.add(word)
        return words

    def getWordLowerSet(self, data_only=True, allow_noncontent_tags=False):
        words = set()

        contentTags = self.tablestore.contentTags

        header = self.header
        cellTags = self.cellTags
        cellWords = self.cellWordsLowerCase
        for cell, tcell, head in zip(cellWords, cellTags, header):
            if not self._isSkipCol(head):
                if not (data_only and self._isFillCol(head)):
                    for alt, talt in zip(cell, tcell):
                        for word, tag in zip(alt, talt):
                            if tag[:2] in contentTags or allow_noncontent_tags:
                                words.add(word)
        return words

    def getPOSSet(self, data_only=True, allow_noncontent_tags=False):
        tags = set()

        contentTags = self.tablestore.contentTags

        header = self.header
        cellTags = self.cellTags
        for cell, head in zip(cellTags, header):
            if not self._isSkipCol(head):
                if not (data_only and self._isFillCol(head)):
                    for alt in cell:
                        for tag in alt:
                            if tag[:2] in contentTags or allow_noncontent_tags:
                                tags.add(tag)
        return tags

    def getLemmaList(self, data_only=True, allow_noncontent_tags=False):
        lemmas = []

        contentTags = self.tablestore.contentTags

        header = self.header
        cellTags = self.cellTags
        cellLemmas = self.cellLemmas
        for cell, tcell, head in zip(cellLemmas, cellTags, header):
            if not self._isSkipCol(head):
                if not (data_only and self._isFillCol(head)):
                    for alt, talt in zip(cell, tcell):
                        for lemma, tag in zip(alt, talt):
                            if tag[:2] in contentTags or allow_noncontent_tags:
                                lemmas.append(lemma)
        return lemmas

    def getWordList(self, data_only=True, allow_noncontent_tags=False):
        words = []

        contentTags = self.tablestore.contentTags

        header = self.header
        cellTags = self.cellTags
        cellWords = self.cellWords
        for cell, tcell, head in zip(cellWords, cellTags, header):
            if not self._isSkipCol(head):
                if not (data_only and self._isFillCol(head)):
                    for alt, talt in zip(cell, tcell):
                        for word, tag in zip(alt, talt):
                            if tag[:2] in contentTags or allow_noncontent_tags:
                                words.append(word)
        return words

    def getWordLowerList(self, data_only=True, allow_noncontent_tags=False):
        words = []

        contentTags = self.tablestore.contentTags

        header = self.header
        cellTags = self.cellTags
        cellWords = self.cellWordsLowerCase
        for cell, tcell, head in zip(cellWords, cellTags, header):
            if not self._isSkipCol(head):
                if not (data_only and self._isFillCol(head)):
                    for alt, talt in zip(cell, tcell):
                        for word, tag in zip(alt, talt):
                            if tag[:2] in contentTags or allow_noncontent_tags:
                                words.append(word)
        return words

    def getPOSList(self, data_only=True, allow_noncontent_tags=False):
        tags = []

        contentTags = self.tablestore.contentTags

        header = self.header
        cellTags = self.cellTags
        for cell, head in zip(cellTags, header):
            if not self._isSkipCol(head):
                if not (data_only and self._isFillCol(head)):
                    for alt in cell:
                        for tag in alt:
                            if tag[:2] in contentTags or allow_noncontent_tags:
                                tags.append(tag)
        return tags

class Table:

    # ============= #
    # Class Methods #
    # ============= #

    def __init__(self, tableName, rows, tablestore):
        self.tableName = tableName
        self.tablestore = tablestore
        self.rows = [TableRow(row, tablestore) for row in rows]

        if len(rows) > 0:
            self.header = rows[0]['header']
        else:
            self.header = []
            
    def __getitem__(self, key):
        return self.rows[key]

    # ============== #
    # Getter Methods #
    # ============== #

    def getHeader(self):
        return self.header

    # ============= #
    # Funct Methods #
    # ============= #

    def isEmpty(self):
        return len(self.rows) == 0

    def isNonEmpty(self):
        return len(self.rows) > 0

class Tablestore:

    # ============= #
    # Class Methods #
    # ============= #

    def __init__(self,
                tablestoreDir: str, 
                cache_dir: str):
        """
            input: Directory that contains the "tables/*.tsv" & the "questions.tsv"
        """

        self.cache_dir = cache_dir

        self.contentTags = ["NN", "VB", "JJ", "RB", "IN", "CD"]

        tablestoreDir = tablestoreDir
        tablestoreTablesDir = f"{tablestoreDir}/tables/"
        questionsAddr = f"{tablestoreDir}/questions.tsv"

        # Verify that the filestructure is formatted correctly
        isErrors = False
        if os.path.exists(tablestoreDir):
            if(os.path.exists(tablestoreTablesDir)):
                if len(os.listdir(tablestoreTablesDir)) > 0:
                    if(os.path.exists(questionsAddr)):

                        self.tablestoreDir = tablestoreDir
                        self.tablestoreTablesDir = tablestoreTablesDir
                        self.questionsAddr = questionsAddr

                        # Set up spacy lemmatizer/tokenizer
                        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #spacy
                        # stanfordnlp.download('en')   #coreNLP
                        # self.nlp = stanfordnlp.Pipeline(lang='en', processors='tokenize,pos,lemma')   #coreNLP

                        # Load the tables
                        tables, tableNames, tableMap = self._load_tables()
                        # self._load_tables_debug()

                        # Load the questions
                        qids, questionsList = self._load_questions()

                        # ==================== #
                        # Misc Precomputations #
                        # ==================== #

                        # Write to class parameters
                        self.tableNames = tableNames
                        self.tables = tables
                        
                        self.uuids = tableMap.keys()
                        self.tableMap = tableMap

                        self.questionIDs = qids
                        self.questionsList = questionsList

                        print("Tablestore fully loaded.\n")
                        print("--------------------------------------\n")
                        
                        self.loaded = True
                    else:
                        print(f"\tERROR! Missing \"questions.tsv\" in tablestore directory \"{tablestoreDir}\".")
                        isErrors = True
                else:
                    print(f"\tERROR! Missing tables in tables/ directory.")
                    isErrors = True
            else:
                print(f"\tERROR! Missing tables/ directory in tablestore directory \"{tablestoreDir}\".")
                isErrors = True
        else:
            print(f"\tERROR! There is no tablestore located at \"{tablestoreDir}\".")
            isErrors = True
     
        if isErrors:
            self.loaded = False
            print("""
                Expected tablestore format:
                    tablestoreDir/
                    tablestoreDir/tables/*.tsv
                    tablestoreDir/questions.tsv
            """)

    # ============== #
    # Getter Methods #
    # ============== #

    def getTableNames(self):
        """
            return:
                List of all table names
        """

        return self.tableNames

    def getTableByName(self, tableName: str):
        """
            Retrieve a table by name

            input:
                tableName - Name of the table
        """
        return self.tables[tableName]

    def getRow(self, uuid: str):
        """
            Get a TableRow for the given UUID
            Will return None if uuid is tempgen

            input:
                uuid - uuid of the tablerow
            return:
                a TableRow
        """
        if uuid in self.tableMap:
            return self.tableMap[uuid]
        else:
            return None
   
    def getRowIDs(self):
        """
            Retrieve a list of all the uuids in the tablestore

            return:
                List of uuids
        """
        return self.tableMap.keys()

    def getQuestionList(self, from_split=[]):
        """
            Get the list of questions, from all  or a specified split of the data

            input:
                from_split - List of splits to include (e.g. ['Dev', 'Test'])
            return:
                List of Question class objects
        """
        questions = self.questionsList
        if len(from_split) == 0:
            return questions
        else:
            return [question for question in questions if question.split in from_split]

    def getQuestion(self, qid: str):
        """
            Retrieve a the question associated with the given id
            (NOTE: Performs a search through a list of questions)

            input:
                qid - Question ID
            return:
                The question object acossiated with the given ID
        """
        questionsList = self.questionsList
        for question in questionsList:
            if question.id == qid : return question
        return None

    def getQuestionsIDs(self):
        """
            Retrieves a list of all question ID's in the tablestore

            retrun:
                list of question IDs
        """
        return self.questionIDs

    # ============= #
    # Funct Methods #
    # ============= #

    def tokenize(self, sentence: str, annotationTypes=['lemma']):
        """
            Tokenizes a given sentence into words, lemmas, or POStags
            Can select any nmber of available options
            If multiple annotation types are selected then this method returns a list of arrays of tokens
            If only one annotation type is selected then this method will return a single array of tokens

            Example:
            tokenize("doggies are animals", annotationTypes=['word', 'lemma']) --> [['doggies', 'are', 'animals'], ['dog', 'is', 'animal']]

            input:
                sentence - Sentence string
                annotationTypes - Annotation requested: 'word', 'lemma', or 'pos'
            return:
                if 1 AnnotationType selected: List of tokens
                else: list of lists of tokens for each annotation type selected 
        """

        # Use spacy to tokenize the sentence
        annotatedSentance = self.nlp(sentence)
        
        # Append each form of annotation requested to a list
        tokens = []
        for annotationType in annotationTypes:
            if annotationType == 'word':
                tokens.append([w.text.lower() for w in annotatedSentance])
            
            elif annotationType == 'lemma':
                tokens.append([w.lemma_.lower() for w in annotatedSentance])
            
            elif annotationType == 'pos':
                tokens.append([w.tag_ for w in annotatedSentance])

            else:
                print(f"\tERROR! Invalid annotation type {annotationType}.")

        # If only one requested return 1d array
        if len(annotationTypes) == 1:
            tokens = tokens[0]

        return tokens

    def _load_tables(self):
        """
            Retrieves the tables from the disk
            Will attempt to read the cache first
            If the cache is old or missing then this function re-reads the tables and annotates them

            return:
                (tables, tableNames, tableMap)
                tables - 
        """


        # location of tables
        tablestoreTablesDir = self.tablestoreTablesDir
        tableFileNames = os.listdir(tablestoreTablesDir)

        # ======= #
        # Loading #
        # ======= #
        
        # Check cache
        tablestore_cached_addr = f"{self.cache_dir}/tablestore_annotated_cache.json"
        refresh_cache = utils.isCacheOld(tablestoreTablesDir, tablestore_cached_addr)

        # If cache old re-load and annotate the tablestore
        if refresh_cache:
            print(f"\tWarning! Cache not up to date, re-annotating tables from \"{tablestoreTablesDir}\"")

            # Get the NLP tool for tokenizing
            nlp = self.nlp

            # For each table make a Table object
            tables = {}     # Dictionary for tableName based lookup
            tableNames = [] # TableNames
            tableMap = {}   # Map for quick lookup by UID
            tableMap_json = {}   # Map for storage on file

            numTables = len(tableFileNames)
            ticker = utils.Ticker(numTables, tick_rate=1, message=f"Annotating {numTables} tables in tablestore...")

            for tableFileName in tableFileNames:
                ticker.tick()
                if tableFileName[-4:] == ".tsv":
                    tableFile_addr = f"{tablestoreTablesDir}/{tableFileName}"
                    tablename = tableFileName[:-4]

                    # Write down tablename
                    tableNames.append(tablename)

                    # read the table file wth pandas
                    try:
                        with open(tableFile_addr) as fp:

                            lines = fp.read().split("\n")

                            header = lines[0].split("\t")
                            header = header[:-1] # fix extra tab in the table files

                            if header[-1] == "[SKIP] UID":

                                # Sanitized header
                                headerSanitized = []
                                for head in header:
                                    head_sanitized = re.sub(r"[^A-Za-z0-9]", " ", head)
                                    head_sanitized = head_sanitized.strip()
                                    head_sanitized = re.sub(r"  +", " ", head_sanitized)
                                    head_sanitized = re.sub(r" ", "_", head_sanitized)

                                    headerSanitized.append(head_sanitized)

                                tableRows = []

                                lines = lines[1:-1] # newline at end of file
                                for line in lines:
                                    cells = line.split("\t")[:-1]

                                    # UUID
                                    uuid = cells[-1]

                                    # Convert the cells into a sentence (skip the skip)
                                    cells_clean = []
                                    for cell, head in zip(cells, header):
                                        if cell != "" and not head.startswith("[SKIP]"):
                                            cell_sanitized = re.sub(r";", " ", cell)
                                            cell_sanitized = cell_sanitized.strip()
                                            cell_sanitized = re.sub(r"  +", " ", cell_sanitized)
                                            cells_clean.append(cell_sanitized)
                                    sentence = " ".join(cells_clean)
                                    tokens = nlp(sentence) #spacy
                                    # tokens = nlp(sentence).sentences[0].words #coreNLP

                                    # Track the token idx to sync the tokens into the 3d cell/alt/word structure
                                    token_idx = 0

                                    # Populate the 3d structures
                                    cellWords = []
                                    cellWordsLowerCase = []
                                    cellLemmas = []
                                    cellTags = []
                                    for cell, head in zip(cells, header):
                                        cell = re.sub(r"  +", " ", cell)        # remove double space typos
                                        isSkipCol = head.startswith("[SKIP]")   # Flag for when its skip column

                                        altWords = []
                                        altWordsLowerCase = []
                                        altLemmas = []
                                        altTags = []

                                        if cell != '':
                                            alts = cell.split(";")
                                            alts = [alt.strip() for alt in alts]

                                            for alt in alts:
                                                word_tokens = nlp(alt) #spacy
                                                # word_tokens = nlp(alt).sentences[0].words #coreNLP

                                                words = []
                                                wordsLowerCase = []
                                                lemmas = []
                                                tags = []

                                                if not isSkipCol:
                                                    for word_token in word_tokens:
                                                        # Spacy
                                                        words.append(tokens[token_idx].text)
                                                        wordsLowerCase.append(tokens[token_idx].lower_)
                                                        lemmas.append(tokens[token_idx].lemma_.lower())
                                                        tags.append(tokens[token_idx].tag_)

                                                        token_idx += 1
                                                        
                                                altWords.append(words)
                                                altWordsLowerCase.append(wordsLowerCase)
                                                altLemmas.append(lemmas)
                                                altTags.append(tags)
                            
                                        # Append an empty alt here
                                        else:
                                            altWords.append([])
                                            altWordsLowerCase.append([])
                                            altLemmas.append([])
                                            altTags.append([])
                                            
                                        cellWords.append(altWords)
                                        cellWordsLowerCase.append(altWordsLowerCase)
                                        cellLemmas.append(altLemmas)
                                        cellTags.append(altTags)

                                    row_obj = {
                                        "uuid": uuid,
                                        "tablename": tablename,
                                        "header": header,
                                        "headerSanitized": headerSanitized,
                                        "cells": cells,
                                        "cellWords": cellWords,
                                        "cellWordsLowerCase": cellWordsLowerCase,
                                        "cellLemmas": cellLemmas,
                                        "cellTags": cellTags,
                                    }

                                    tableRow = TableRow(row_obj, self)

                                    tableRows.append(row_obj)

                                    tableMap[uuid] = tableRow

                                table = Table(tablename, tableRows, self)
                                
                                tableMap_json[tablename] = tableRows
                                
                                # Safty check (dont keep empty tables)
                                if table.isNonEmpty:
                                    tables[tablename] = table
                                else:
                                    tableNames.remove(tablename)

                            else:
                                raise IOError
                    except IOError:
                        print('\n', end='\r', flush=True)
                        print(f"\tWarning! Cannot parse table file \"{tableFile_addr}\".")

            ticker.end()

            print(f"\tCaching annotated tables to \"{tablestore_cached_addr}\".")
            with open(tablestore_cached_addr, mode='w') as fp:
                fp.write(json_dumps(tableMap_json))

            return tables, tableNames, tableMap

        else:
            print(f"\tLoading cached annotated table from \"{tablestore_cached_addr}\".")
            if os.path.exists(tablestore_cached_addr):
                with open(tablestore_cached_addr, mode='r') as fp:

                    # Parse the JSON file
                    tablestoreJSON = json.load(fp)

                    # The keys will be each tableName
                    tableNames = tablestoreJSON.keys()

                    # For each table make a Table object
                    tables = {}     # Dictionary for tableName based lookup
                    tableMap = {}   # Map for quick lookup by UID
                    for tableName in tableNames:
                        rows = tablestoreJSON[tableName]
                        table = Table(tableName, rows, self)

                        # Safty check (dont keep empty tables)
                        if table.isNonEmpty:
                            tables[tableName] = table

                            # Add the rows to the tableMap
                            for row in table.rows:
                                uuid = row.uuid
                                tableMap[uuid] = row
                        else:
                            tableNames.remove(tableName)

                    return tables, tableNames, tableMap  

            else:
                print(f"\tERROR! Failed to load tablestore from JSON \"{tablestore_cached_addr}\"")
                return False, False, False

    def _load_questions(self):

        questionsAddr = self.questionsAddr
        questions_cached_addr = f"{self.cache_dir}/questions_annotated_cache.json"

        # Question map
        questionsList = []
        questionsList_JSON = [] # Parallel structure that doesnt use classes so that we can cache
        qids = []

        # check to see if cached questions are on file
        refresh_cache = utils.isCacheOld(questionsAddr, questions_cached_addr)
        if refresh_cache:

            print(f"Loading questions from \"{questionsAddr}\".")

            # Load the questions table
            questions = pd.read_csv(questionsAddr, sep="\t")

            # Filter out incomplete questions
            questions = questions[questions['flags'].notna()]
            questions = questions[questions['explanation'].notna()]
            questions = questions[questions['examName'].notna()]
            questions = questions[questions.flags.str.contains("SUCCESS|READY", regex=True)]

            # Annotate the questions and save them locally

            # Prepare interface for user to see progress
            numQs = questions.shape[0]
            ticker = utils.Ticker(numQs, tick_rate=166, message=f"Annotating {numQs} questions...")

            # Convert questions to a map
            # store the question dataframe; the lemmatized quesdtion text; the lemmatized answer text
            for i, question_df in questions.iterrows():
                question = question_df.to_dict()

                qid = question['QuestionID']
                qids.append(qid)

                # info for user
                ticker.tick()
                
                # Read the question text and split it up into question and all answers
                qText = question['question']
                qText = re.split(r"\([A-G1-7]\)", qText)

                # Distinguish Q and A
                q_text, q_anss = qText[0], qText[1:]

                # Remove outer whitespace
                q_text = q_text.strip()
                q_anss = [ans.strip() for ans in q_anss]
                
                # Use spacy to annotate the texts

                # Question text annotation
                [q_text_words, q_text_lemmas, q_text_pos] = self.tokenize(q_text, annotationTypes=['word', 'lemma', 'pos'])
                
                # Answer text annotation
                isAlphaAnswerKey = re.match(r'[A-Ga-g]', question['AnswerKey'])
                answerSelections = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] if isAlphaAnswerKey else ['1', '2', '3', '4', '5', '6', '7']
                answers_annotation = {}
                for i_ans, q_ans in enumerate(q_anss):
                    [ans_text_words, ans_text_lemmas, ans_text_pos] = self.tokenize(q_ans, annotationTypes=['word', 'lemma', 'pos'])
                    answers_annotation[answerSelections[i_ans]] = {
                        "words": ans_text_words,
                        "lemmas": ans_text_lemmas,
                        "pos": ans_text_pos
                    }

                # Add the annotation to the question
                question["question_annotation"] = {
                    "words": q_text_words,
                    "lemmas": q_text_lemmas,
                    "pos": q_text_pos
                }

                question["answers_annotation"] = answers_annotation

                questionsList_JSON.append(question)

                # Convert to a question object
                question = Question(question)

                # Add to the list
                questionsList.append(question)

            ticker.end()

            # Cache the annotated questions locally
            print(f"\tCaching annotated questions to \"{questions_cached_addr}\".")
            with open(questions_cached_addr, mode="w") as fp:
                fp.write(json_dumps(questionsList_JSON))
        else:
            print(f"\tLoading cached annotated questions from \"{questions_cached_addr}\".")
            with open(questions_cached_addr) as fp:
                questionsList = json_load(fp)
                
                # Convert to question class
                questionsList = [Question(q) for q in questionsList]

                # Get the list of qids
                qids = [q.id for q in questionsList]

        return qids, questionsList

def loadTablestore(tablestoreDir, cache_dir):

    tablestore = Tablestore(tablestoreDir, cache_dir=cache_dir)

    if tablestore.loaded: 
        return tablestore
    else:
        print(f"\tERROR! Failed to load tablestore from directory \"{tablestoreDir}\".")
        print(f"Require \"questions.tsv\" and \"tables/*.tsv\" files.")
        None

if __name__ == "__main__":

    TABLESTORE_DIR = "data/tablestore/tablestore-2020-04-26/"
    CACHE_DIR      = "cache/"
    
    # Load the tablestore
    tablestore = loadTablestore(TABLESTORE_DIR, CACHE_DIR)

    if tablestore:
        # read the rows

        # read the questions
        uuids = tablestore.getRowIDs()
        for uuid in uuids:
            
            tableRow = tablestore.getRow(uuid)

            data_words = tableRow.getWordList(data_only=True)
            all_words = tableRow.getWordList(data_only=False)

            pass
