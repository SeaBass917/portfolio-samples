///////////////////////////////////////////////////////////////////////////////////////////////////
// Authors:         Sebastian Thiem, Peter Jansen
//
// Project Name:    Syncronicity
// File Name:		server.js
// Create Date:     11 September 2019
//
// Description:     Back-end to the Inference Pattern authoring tool 
//                  Syncronicity
//
///////////////////////////////////////////////////////////////////////////////////////////////////
'use strict';

/* 
 * Include Statements
 */
const execSync = require('child_process').execSync;
var async = require('async');
var express = require("express");
var _ = require("lodash");
var bodyParser = require('body-parser');
var fs = require('fs');
var dateTime = require('node-datetime');
var url = require('url');
//var GoogleSpreadsheet = require('./index.js');				// New V4 Sheets AP

var spawn = require('child_process').spawn;

var tablestore = require('./tablestore.js');                                    // Tablestore
var tablestoreGoogleSheets = require('./tablestoreGoogleSheets.js');            // Tablestore

// Set to 'true' to enable CoreNLP annotation for the Tablestore using the external Scala tool.
// Set to 'false' to disable this for speed/debugging. 
var CORENLP_ANNOTATION_ENABLED  = true; 
var DEBUG_LOAD_PREANNOTATED_TABLESTORE  = true; 


/*
 * Express Configurations
 */
var app = express();
app.use("/", express.static(__dirname, +"/www"));
app.use("/css", express.static(__dirname + "/node_modules/ui-bootstrap/vendor/ui-bootstrap"));
app.use("/js", express.static(__dirname + "/node_modules/angular"));
app.use("/js", express.static(__dirname + "/js"));
app.use("/imlrun", express.static(__dirname, +"/runscript/output"));
app.use("/monaco-editor", express.static(__dirname + "/node_modules/monaco-editor"));
app.use(bodyParser.json({
    limit: '50mb' // Use this to extend bandwidth from the server
}))
app.use(bodyParser.urlencoded({ // NOTE order matters .json() -> .urlencoded()
    limit: '50mb',  // Use this to extend bandwidth to the server
    extended: true,
}));

/*
 * Server Variables/Data 
 */
var PORT = 8080;            // HTTP port for server (usually 80 or 8080 -- must match client)
var ANNOTATION_PATH = "../annotation/"
var PATTERNS_PATH = `${ANNOTATION_PATH}patterns/`;     // Location for data files
var QUESTIONS_PATH = `${ANNOTATION_PATH}expl-tablestore-export-2019-11-22-173702/`
var TABLESTORELIVE_PATH = `${ANNOTATION_PATH}tablestore/`
var ROW_ROLE_FREQS_PATH = "data/rowRoleFreqs.JSON"
var LEMMA_PATH = 'data/lemmatization-en.txt'



// Start listening
var server = app.listen(PORT, function () {
    var host = server.address().address;
    var port = server.address().port;
    console.log("App listening at http://%s:%s", host, port);
});

// Hashmap that enumerates words based on the local lemmatizer file
var lemmatizer = {}
let lemmatizerReady = false
loadLemmatizer()

// Globals
var selectedDataset = ''

var currentStatistics = { isValidQ: false, isValidT: false };

/*
 * Tablestore/Google Sheets interface variables
 */

var rows = {};
var rowsAnnotated = [];
var tablestoreAnnotated = {};
var questions = [];

/*
 * Pattern generation variables
 */


// Map of all role frequencies for each row
var rowRoleFreqs = {}
loadRowRoleFreqs()

// Read in and interperet the labels and grids available
var possibleDatasetDirs = []
var possibleDatasets = []
var selectedDatasetIdx = -1
var selectedDataset = ""
var possiblePatternFiles = []
var possiblePatterns = []
var selectedPatternIdx = -1
var selectedPattern = ""
getDataSets()

// Main initialization
//## tablestoreGoogleSheets.loadTablestoreRowsFromGoogleSheets();
//## initialize();

/*
 * GET Requests
 */

// Give the client a list of dataset options
app.get("/GetDatasetSelections", function (req, res) {
    console.log("GET request for the DatasetSelections")

    // Load the preferences
    var prefs = getPrefs()

    var data = {
        possibleDatasets: possibleDatasets,
        prefs: prefs,
    }

    res.status(200).send(data)
})

// Give the client a list of pattern options based on the current dataset
app.get("/GetPatternSelections", function (req, res) {
    console.log("GET request for the patternSelections")

    var data = {
        possiblePatterns: possiblePatterns,
    }

    res.status(200).send(data)
})

// Send the current dataset and pattern selected to the client
app.get("/GetDatasetAndPattern", function (req, res) {

    var data = {
        selectedDataset: selectedDataset,
        selectedPattern: selectedPattern,
    }

    res.status(200).send(data)
})

// Give the client the patterns on the server
app.get("/GetPatterns", function (req, res) {

    console.log(req.query)

    selectedDataset = req.query.selectedDataset

    // Ensure that there exists an IML and edge tables director in the dataset
    if(!fs.existsSync(`${PATTERNS_PATH}/${selectedDataset}/IML_manual/`)) execSync(`mkdir ${PATTERNS_PATH}/${selectedDataset}/IML_manual/`)
    if(!fs.existsSync(`${PATTERNS_PATH}/${selectedDataset}/IML_auto/`)) execSync(`mkdir ${PATTERNS_PATH}/${selectedDataset}/IML_auto/`)
    if(!fs.existsSync(`${PATTERNS_PATH}/${selectedDataset}/edge_tables/`)) execSync(`mkdir ${PATTERNS_PATH}/${selectedDataset}/edge_tables/`)
    
    let data = loadPatterns(selectedDataset)
    data['edgeTablesMap'] = loadEdgeTablesMap(selectedDataset)

    let patternsPresent = data.hasOwnProperty('patterns')
    let patternNamesPresent = data.hasOwnProperty('patternNames')

    if (patternsPresent && patternNamesPresent) {
        res.status(200).send(data)
    }
    else {
        if (!patternsPresent) {
            console.log('\tERROR! Failed missing prop "patterns" in pattern request.')
        }
        if (!patternNamesPresent) {
            console.log('\tERROR! Failed missing prop "patternNamesPresent" in pattern request.')
        }
        res.status(500).send('Datastructure issue.')
    }
})

// ### TODO change this to just done when the patterns are complete
// Give the client the patterns marked as done on the server
app.get("/GetPatternsDone", function (req, res) {

    console.log(req.query)

    selectedDataset = req.query.selectedDataset

    let _data = loadPatterns(selectedDataset)

    let patternsPresent = _data.hasOwnProperty('patterns')
    let patternNamesPresent = _data.hasOwnProperty('patternNames')

    if (patternsPresent && patternNamesPresent) {

        // Filter down to just the Done and good patterns
        let data = {
            patterns: [],
            patternNames: [],
        }

        let numPatterns = _data.patterns.length
        for (let i = 0; i < numPatterns; i++) {
            let pattern = _data.patterns[i]
            let patternName = _data.patternNames[i]

            if (pattern.isDone || pattern.isGood) {
                data.patterns.push(pattern)
                data.patternNames.push(patternName)
            }
        }

        res.status(200).send(data)
    }
    else {
        if (!patternsPresent) {
            console.log('\tERROR! Failed missing prop "patterns" in pattern request.')
        }
        if (!patternNamesPresent) {
            console.log('\tERROR! Failed missing prop "patternNames" in pattern request.')
        }
        res.status(500).send('Datastructure issue.')
    }
})

// Give the client the edgeTables on the server
app.get("/GetEdgeTables", function (req, res) {

    console.log(req.query)

    selectedDataset = req.query.selectedDataset

    let data = loadEdgeTables(selectedDataset)

    let edgeTablesPresent = data.hasOwnProperty('edgeTables')
    let edgeTableNamesPresent = data.hasOwnProperty('edgeTableNames')

    if (edgeTablesPresent && edgeTableNamesPresent) {
        res.status(200).send(data)
    }
    else {
        if (!edgeTablesPresent) {
            console.log('\tERROR! Failed missing prop "edgeTables" in edgeTable request.')
        }
        if (!edgeTableNamesPresent) {
            console.log('\tERROR! Failed missing prop "edgeTableNames" in edgeTable request.')
        }
        res.status(500).send('Datastructure issue.')
    }
})

// Give the client a lemmatizer map
app.get("/GetLemmatizer", function (req, res) {

    if (lemmatizerReady) {
        res.status(200).send(lemmatizer)
    }
    else {
        // Lemmatizer not fully loaded
        res.status(204).send({})
    }
})

// Request from the client for the Tablestore. 
// Fetch Tablestore from Google Sheets, pack, and send to client. 
var numAttempts = 0;
app.get("/GetTablestore", function (req, res) {

    // Load the preferences
    var prefs = getPrefs()

    // Begin downloading the tablestore from Google Sheets    
    tablestoreGoogleSheets.loadTablestoreRowsFromGoogleSheets();                                            // Get tablestore in row format
    if (CORENLP_ANNOTATION_ENABLED == true) {
       tablestoreGoogleSheets.loadAndExportTablesFromGoogleSheets(TABLESTORELIVE_PATH, function () {           // Get tablestore and export it
            tablestoreGoogleSheets.annotateExportedTables(prefs.worldtreeAddress, TABLESTORELIVE_PATH, DEBUG_LOAD_PREANNOTATED_TABLESTORE, function() {                     // Then annotate the tablestore

            });
        });
    }

    numAttempts = 0;
    // Start checking if data is ready (and, send it when ready)
    sendWhenReady();

    function sendWhenReady() {
        var maxAttempts = 60;
        console.log("rowsLoaded: " + tablestoreGoogleSheets.checkIfRowsLoaded() + "   rowsAnnotated: " + tablestoreGoogleSheets.checkIfRowsAnnotated())        

        if ((CORENLP_ANNOTATION_ENABLED == true && tablestoreGoogleSheets.checkIfRowsLoaded() && tablestoreGoogleSheets.checkIfRowsAnnotated()) ||
            (CORENLP_ANNOTATION_ENABLED == false && tablestoreGoogleSheets.checkIfRowsLoaded())) {
            rows = tablestoreGoogleSheets.getLoadedRows();
            rowsAnnotated = tablestoreGoogleSheets.getAnnotatedRows();
            tablestoreAnnotated = tablestoreGoogleSheets.getAnnotatedTablestore();
            let tableHeaders = tablestoreGoogleSheets.getLoadedHeaders();

            console.log("Server: Rows returned: " + rows.length)

            // Load questions and explanations
            questions = tablestore.loadQuestions(`${QUESTIONS_PATH}/ARC-QC+EXPL-Train-1.tsv`, rows);

            // Send data to client
            res.status(200).send({
                rows: rows,
                rowsAnnotated: rowsAnnotated,
                questions: questions,
                rowRoleFreqs: rowRoleFreqs,
                lemmatizer: lemmatizer,
                tableHeaders: tableHeaders,
            })

        } else {
            numAttempts += 1;
            console.log("Checking if tablestore data has been retrieved (numAttempts: " + numAttempts + ")")
            if (numAttempts < maxAttempts) {
                setTimeout(sendWhenReady, 1000)
            } else {
                console.log("ERROR: Tablestore was not ready to send to client and exceeded the maximum number of attempts (" + maxAttempts + "). ")
            }
        }
    }    

});

/*
 * POST Requests
 */

// Allow the client to change the current label
app.post('/PostDatasetSelection', function (req, res) {

    // Update the current label and label idx
    selectedDataset = req.body.selectedDataset
    selectedDatasetIdx = getDatasetIdx(selectedDataset)

    console.log(`Dataset Selection changed to: "${selectedDataset}"`)

    // Refresh the possible patterns for new dataset
    console.log('Reading Pattern Selections')
    updatePatternSelections()

    // Respond with the data selection state
    var data = {
        selectedDataset: selectedDataset,
        selectedPattern: selectedPattern,
    }

    // console.log(`Selected: {
    //     selectedDataset: ${selectedDataset},
    //     selectedPattern: ${selectedPattern},
    // }`)

    res.status(200).send(data)
})

// Allow the client to change the current pattern
app.post('/PostPattern', function (req, res) {

    let data = JSON.parse(req.body.dataString)

    // Unpackage the request
    let pattern = data.pattern
    let patternName = data.patternName
    let edgeTable = data.edgeTable
    let patternIML = data.patternIML
    let autoGenIML = data.autoGenIML
    let dataset = data.dataset
    let newName = data.newName
    let copy = data.copy

    let imlEditsPresent = patternIML != autoGenIML

    // ----------------------
    // - Save Pattern Frame -
    // ----------------------

    // Stringify the pattern dataframe into a tsv
    let patternStr = stringifyDF(pattern, "\t", true)

    // Create the filepath string, use the 'new name' if there is one
    let isNewName = newName.length > 0
    let filePath = (isNewName) ? `${PATTERNS_PATH}${dataset}/${newName}.tsv` : `${PATTERNS_PATH}${dataset}/${patternName}.tsv`;
    console.log(`writing pattern to "${filePath}"`)

    // Save the pattern 
    let errorMessages = []
    fs.writeFile(filePath, patternStr, function (err) {
        if (err) {
            console.log(`\tError! Failed to save pattern: "${filePath}" to local storage.`)
            console.log(err)
            res.status(500).send(`\tError! Failed to save pattern: "${filePath}" to local storage.`)
        }
        else {
            console.log(`Saved pattern: "${patternName}" to "${filePath}"`)

            // If there is a new name, but we are not making a copy, then we are overwritting the previous pattern
            // delete(unlink) the prev pattern from the fs
            if (isNewName && !copy) {

                // Point to the old fileName
                filePath = PATTERNS_PATH + `${dataset}/${patternName}.tsv`

                // Delete it
                fs.unlink(filePath, function (err) {
                    if (err) {
                        console.log(`\tError! Failed to delete pattern: "${filePath}" from local storage.`)
                        errorMessages.push(`\tError! Failed to delete pattern: "${filePath}" from local storage.`)
                    }
                    else console.log(`Deleting pattern: "${filePath}"`)
                })
            }

            // Save the edge table and IML (only if the pattern is marked as done or good)
            if (pattern.isDone || pattern.isGood) {

                // -------------------
                // - Save Edge Table -
                // -------------------

                // Edge tables have prefix
                let edgeTableName = `edges_${patternName}`
                let newEdgeTableName = `edges_${newName}`

                // Create the filepath string, use the 'new name' if there is one
                filePath = (isNewName) ? `${PATTERNS_PATH}${dataset}/edge_tables/${newEdgeTableName}.tsv` : `${PATTERNS_PATH}${dataset}/edge_tables/${edgeTableName}.tsv`;
                console.log(`writing to edge table "${filePath}"`)

                let edgeTableStr = stringifyDF2(edgeTable, "\t")

                // Save the edgeTables 
                fs.writeFile(filePath, edgeTableStr, function (err) {
                    if (err) {
                        console.log(`\tError! Failed to save edge table: "${filePath}" to local storage.`)
                        console.log(err)
                        res.status(500).send(`\tError! Failed to save edge table: "${filePath}" to local storage.`)
                    }
                    else {
                        console.log(`Saved edge table: "${edgeTableName}" to "${filePath}"`)

                        // If there is a new name, but we are not making a copy, then we are overwritting the previous pattern
                        // delete(unlink) the prev pattern from the fs
                        if (isNewName && !copy) {

                            // Point to the old fileName
                            filePath = `${PATTERNS_PATH}${dataset}/edge_tables/${edgeTableName}.tsv`

                            // Delete it
                            fs.unlink(filePath, function (err) {
                                if (err) {
                                    console.log(`\tError! Failed to delete edge_table: "${filePath}" from local storage.`)
                                    errorMessages.push(`\tError! Failed to delete edge_table: "${filePath}" from local storage.`)
                                }
                                else console.log(`Deleting pattern: "${filePath}"`)
                            })
                        }

                        // -----------------
                        // - Save IML_auto -
                        // -----------------

                        // Create the filepath string, use the 'new name' if there is one
                        if(isNewName) filePath = `${PATTERNS_PATH}${dataset}/IML_auto/${newName}.iml`
                        else filePath = `${PATTERNS_PATH}${dataset}/IML_auto/${patternName}.iml`
                        console.log(`writing to IML_auto "${filePath}"`)

                        // Save the IML 
                        fs.writeFile(filePath, autoGenIML, function (err) {
                            if (err) {
                                console.log(`\tError! Failed to save IML: "${filePath}" to local storage.`)
                                console.log(err)
                                res.status(500).send(`\tError! Failed to save IML: "${filePath}" to local storage.`)
                            }
                            else {
                                console.log(`Saved IML: "${patternName}" to "${filePath}"`)

                                // If there is a new name, but we are not making a copy, then we are overwritting the previous pattern
                                // delete(unlink) the prev pattern from the fs
                                if (isNewName && !copy) {

                                    // Note here we need to delete arny reference to this pattern in either folder

                                    // Point to the old fileName
                                    filePath = `${PATTERNS_PATH}${dataset}/IML_auto/${patternName}.iml`

                                    // Delete it
                                    fs.unlink(filePath, function (err) {
                                        if (err) {
                                            console.log(`\tError! Failed to delete iml: "${filePath}" from local storage.`)
                                            errorMessages.push(`\tError! Failed to delete iml: "${filePath}" from local storage.`)
                                        }
                                        else console.log(`Deleting iml: "${filePath}"`)
                                    })
                                }

                                // -------------------
                                // - Save IML_manual -
                                // -------------------
                                // Only if there are maunal changes to be saved
                                if(imlEditsPresent){
                                    if(isNewName) filePath = `${PATTERNS_PATH}${dataset}/IML_manual/${newName}.iml`
                                    else filePath = `${PATTERNS_PATH}${dataset}/IML_manual/${patternName}.iml`
                                    console.log(`writing to IML_manual "${filePath}"`)

                                    // Save the IML 
                                    fs.writeFile(filePath, patternIML, function (err) {
                                        if (err) {
                                            console.log(`\tError! Failed to save IML: "${filePath}" to local storage.`)
                                            console.log(err)
                                            res.status(500).send(`\tError! Failed to save IML: "${filePath}" to local storage.`)
                                        }
                                        else {
                                            console.log(`Saved IML: "${patternName}" to "${filePath}"`)
            
                                            // If there is a new name, but we are not making a copy, then we are overwritting the previous pattern
                                            // delete(unlink) the prev pattern from the fs
                                            if (isNewName && !copy) {
            
                                                // Note here we need to delete arny reference to this pattern in either folder
            
                                                // Point to the old fileName
                                                filePath = `${PATTERNS_PATH}${dataset}/IML_manual/${patternName}.iml`
            
                                                // Delete it
                                                fs.unlink(filePath, function (err) {
                                                    if (err) {
                                                        console.log(`\tError! Failed to delete iml: "${filePath}" from local storage.`)
                                                        errorMessages.push(`\tError! Failed to delete iml: "${filePath}" from local storage.`)
                                                    }
                                                    else console.log(`Deleting iml: "${filePath}"`)
                                                })
                                            }
            
                                            res.status(200).send(`Saved pattern: "${patternName}" under dataset "${dataset}".`)
                                        }
                                    })


                                }
                                else{
                                    res.status(200).send(`Saved pattern: "${patternName}" under dataset "${dataset}".`)
                                }
                            }
                        })
                    }
                })
            }
            else {
                res.status(200).send(`Saved pattern: "${patternName}" under dataset "${dataset}".`)
            }
        }
    })
})

// Allow the client to change the current pattern
app.post('/PostEdgeTable', function (req, res) {

    let data = req.body

    // Unpackage the request
    let edgeTable = data.edgeTable
    let edgeTableName = data.edgeTableName
    let dataset = data.dataset

    // Stringify the edge Table dataframe into a tsv
    let edgeTableStr = stringifyDF2(edgeTable, "\t")

    // Create the filepath string
    let filePath = `${PATTERNS_PATH}/${dataset}/edge_tables/${edgeTableName}.tsv`
    console.log(`writing to "${filePath}"`)

    // Save the pattern 
    fs.writeFile(filePath, edgeTableStr, function (err) {
        if (err) {
            console.log(`\tError! Failed to save pattern: "${filePath}" to local storage.`)
            console.log(err)
            res.status(500).send(`\tError! Failed to save pattern: "${filePath}" to local storage.`)
        }
        else {
            console.log(`Saved pattern: "${edgeTableName}" to "${filePath}"`)
            res.status(200).send(`Saved pattern: "${edgeTableName}" under dataset "${dataset}".`)
        }

        let output = execSync(`git add ${PATTERNS_PATH}`, { encoding: 'utf-8' });
        console.log("-------------------------------------")
        console.log(output)
        console.log("-------------------------------------")
    })
})

// Allow the client to change the current pattern
app.post('/PostPublish', function (req, res) {
    console.log("* PostPublish(): Started... ")

    // For holding the console output
    let output = ""

    // What will be returned
    let message = ''
    let patternsNew = null
    let mergeConflicts = false

    // Commitiing on an unchanged branch throws an exception
    // so if it failes, just catch that and move on

    console.log("* PostPublish(): Committing... ")
    try {
        // Commit
        output = execSync(`git add ${PATTERNS_PATH}`, { encoding: 'utf-8' })
        output = execSync('git commit -am "Data (In-app commit)"', { encoding: 'utf-8' })
        message += "Data Committed. "
    }
    catch (err) {
        output = err.output[1]  // Note function call never returned , so output is found here

        if (output.indexOf('Your branch is up to date') >= 0) {
            message += "No data to commit. "
        }
        else {
            message += "ERROR! Failed to commit for unknown reason. Check server."
            res.status(500).send(output)
            return
        }
    }

    console.log("* PostPublish(): Pulling... ")
    // Pull changes throws exception for: already up to date and merge conflicts
    try {
        output = execSync('git pull', { encoding: 'utf-8' })
        message += "Data Pulled. "

        // Not up to date, means a merge happened, but if no exception was thrown it was a good merge
        // load the changes
        if (output.indexOf('Already up to date.') < 0) {
            patternsNew = loadPatterns(selectedDataset)
            message += "Changes were made externally, reading changes now. "
        }

    } catch (err) {
        output = err.output[1]  // Note function call never returned , so output is found here

        // If something was pulled then we need to read the changes and send them to the user
        if (output.indexOf('Already up to date.') < 0) {

            // If the merging was fine load the patterns
            if (output.indexOf('Automatic merge failed') >= 0) {
                mergeConflicts = true
                message += "ERROR! Merge conflicts. "
            }
        }
    }

    console.log("* PostPublish(): Pushing... ")
    // Push our changes (only if there were no conflicts)
    if (!mergeConflicts) {
        output = execSync('git push', { encoding: 'utf-8' })
        message += "Data Pushed.";
    }
    else {
        message += "Cannot push with merge conflicts."
    }

    console.log("* PostPublish(): Sending... ")
    res.status(200).send({
        update: patternsNew,
        message: message,
        mergeConflicts: mergeConflicts,
    })

    console.log("* PostPublish(): Completed... ")
})

// Interface for seeing if the server is up
app.get('/Ping', function (req, res) {
    res.status(200).send('.')
})

// Exports the current dataset as a zip
app.post('/Export', function (req, res) {

    // Export a fresh copy of the tablestore
    //tablestoreGoogleSheets.loadAndExportTablesFromGoogleSheets(TABLESTORELIVE_PATH, function () {
    tablestoreGoogleSheets.loadAndExportTablesFromGoogleSheets(TABLESTORELIVE_PATH, function () {        

        // TODO: After tablestore loaded/exported. 
        // Get a date string to tag the export
        let date = getDate()

        // Zip the annotation folder up (isolate only the dataset we are interested in)
        let output = execSync(`tar -czvf ${selectedDataset}_${date}.tar.gz ${ANNOTATION_PATH}patterns/${selectedDataset}/ ${QUESTIONS_PATH} ${TABLESTORELIVE_PATH}`, { encoding: 'utf-8' })
        console.log(`$ tar -czvf ${selectedDataset}_${date}.tar.gz ${ANNOTATION_PATH}patterns/${selectedDataset}/ ${QUESTIONS_PATH} ${TABLESTORELIVE_PATH}`)
        console.log(output)

        // Move the zipped package into the annotation folder
        output = execSync(`mv ${selectedDataset}_${date}.tar.gz ${ANNOTATION_PATH}${selectedDataset}_${date}.tar.gz`, { encoding: 'utf-8' })
        console.log(`$ mv ${selectedDataset}_${date}.tar.gz ${ANNOTATION_PATH}${selectedDataset}_${date}.tar.gz`)
        console.log(output)

        //## Test
        //tablestoreGoogleSheets.annotateExportedTables(TABLESTORELIVE_PATH)


        res.status(200).send()        
    });

})

// Exports the current dataset as a zip
app.post('/RunPattern', function (req, res) {
    // Load the preferences
    var prefs = getPrefs()
    const path = prefs.worldtreeAddress           // PATH TO IML INTERPRETER

    console.log("Started...")
    
    // Check that worldtree is present before running
    if(fs.existsSync(path)){

        var runscriptPath = "runscript";

        console.log("RunPattern(): started... ");

        // Unpack request data (to extract filenames)
        let patternName = req.body.patternName
        let dataset = req.body.dataset
        let timeLimit = req.body.timeLimit;        

        // Determine if there is a manual IML on file, if not switch to the auotmatically generated file
        var imlFilename = `${PATTERNS_PATH}${dataset}/IML_manual/${patternName}.iml`
        if(!fs.existsSync(imlFilename)) imlFilename = `${PATTERNS_PATH}${dataset}/IML_auto/${patternName}.iml`

        // Make sure we are pointing to an IML file now
        if(fs.existsSync(imlFilename)){

            //var tableMatchesFilename = "runscript/output/infpatdebugexport-" + patternName.replace(/[^A-Za-z0-9]/g, "") + ".html"
            var tableMatchesFilename = "runscript/output/infpatdebugexport.html"
            var tableMatchesHTMLStr = "";
            tableMatchesHTMLStr += "<html><head><meta http-equiv=\"refresh\" content=\"1\"></head><body><font color=\"grey\">Table matches analysis will automatically refresh when data is available.</font></html>";
            
            fs.writeFile(tableMatchesFilename, tableMatchesHTMLStr, function(err) {
                if (err) {
                    console.log("Warning: Unable to reset table matches file: " + tableMatchesFilename)
                }
            });
            
    
            // Step 1A: Export IML for specified pattern
            // Note, this should already be done above
            //var imlFilename = "imltest.iml";
    
            // Step 1B: Export IML run script for that IML pattern (that includes the above script)
    
            safeExecSync(`mkdir ${runscriptPath}`);
            safeExecSync(`mkdir ${runscriptPath}/output/`);
    
            var runscriptFilename = runscriptPath + "/runscript.iml";
    
            var runscriptIMLStr = "";
            runscriptIMLStr += "// This file is an automatically generated runscript from the inference pattern generation tool. \n";
            runscriptIMLStr += "import \"../" + imlFilename + "\"" + "\n";
            runscriptIMLStr += "\n";
            runscriptIMLStr += '// Perform constraint satisfaction over the inference pattern and generate the HTML logfile output. \n'
            runscriptIMLStr += "executeAutoPatterns\n";
            runscriptIMLStr += "exportInfPatDebugHTML()\n"
            runscriptIMLStr += "populateInfPatMatches\n";            
            runscriptIMLStr += "incrementState\n";
            runscriptIMLStr += "exportInfPatHTML()\n";
            runscriptIMLStr += "exportTableStoreHTML()\n";
            runscriptIMLStr += "exportStateSpaceHTML()\n";
    
            fs.writeFile(runscriptFilename, runscriptIMLStr, function (err) {
                if (err) {
                    console.log("\tERROR: Failed to generate IML run script. ")
                    console.log(err)
                    res.status(500).send(`\tERROR: Failed to generate IML run script.`)
                }
            });
    
            // Clear old output logfile
            fs.writeFile(runscriptPath + "/output/runscript.infpat.html", "", function (err) {
                if (err) {
                    /*
                    console.log("\tERROR: Failed to generate IML run script. ")
                    console.log(err)
                    res.status(500).send(`\tERROR: Failed to generate IML run script.`)
                    */
                }
            });
    
            /*
            // NOTE: Disabling this as the most-recent version of the tablestore should already be on disk, and this needlessly hits the Google API (which may be subject to throttling)
            // Step 2: Export tablestore    
            tablestoreGoogleSheets.loadAndExportTablesFromGoogleSheets(TABLESTORELIVE_PATH, function () {
                // TODO: After tablestore loaded/exported. 
            });
            */
    
    
            // Step 3: Run IML Interpreter
    
            // Path information
            var SharedData = {};    
    
            SharedData.ProjectPath = "";
    
            /*
            var scriptFilename = SharedData.ProjectPath + "before10temp2/scripts/runme/" + "runme" + ".iml";
            var outputPath = SharedData.ProjectPath + "before10temp2/output/";
            var tablestoreIndex = SharedData.ProjectPath + "annotation/expl-tablestore-export-2020-01-16-163150/" + "tableindex.txt";
            */
            var scriptFilename = __dirname + "/" + runscriptFilename;
            var outputPath = __dirname + "/" + runscriptPath + "/output/";
            //var tablestoreIndex = SharedData.ProjectPath + "annotation/expl-tablestore-export-2020-01-16-163150/" + "tableindex.txt";
            var tablestoreIndex = __dirname + "/" + TABLESTORELIVE_PATH + "tableindex.txt"
    
    
            // Spawn interpreter    
            const cmd = "sbt"
            const args = ["-J-Xmx4g", "\"runMain inferenceengine.iml.runtime.IMLInterpreterCmdLine",
                "--script", scriptFilename,
                "--tablestoreIndex", tablestoreIndex,
                "--outputPath", outputPath,
                "--timeLimit", timeLimit,
                "\""]
            const spawnConfig = {
                cwd: path,                      // Current working directory
                env: {                          // Environment variables
                    PATH: process.env.PATH,
                    SBT_OPTS:"-Xmx4g"
                },
                shell: true
            }
            
            const p = spawn(cmd, args, spawnConfig)
    
            var consoleOutput = [];
    
            // Listen for output
            p.stdout.on('data', (data) => {
                console.log('stdout: ' + data)
                //this.console.text = "stdout: " + data;
    
                var consoleInput = data.toString().split("\n");
                for (var lineIdx in consoleInput) {
                    //this.addConsoleOutputToBuffer(consoleInput[lineIdx]);
                    consoleOutput.push(consoleInput[lineIdx]);
                }
    
                /*
                p.on('close', (code) => {
                    //this.refreshLogStateSpace();
                    this.addConsoleOutputToBuffer('child process exited with code' + code.toString());
                })
                */
            });
    
            // Listen for errors
            p.stderr.on('data', (data) => {
                console.log('stderr: ' + data)
                //this.console.text = "stderr: " + data;
    
                var consoleInput = data.toString().split("\n");
                for (var lineIdx in consoleInput) {
                    //this.addConsoleOutputToBuffer(consoleInput[lineIdx]);
                    consoleOutput.push(consoleInput[lineIdx]);
                }
    
            });
    
            p.on('close', (code) => {
                console.log('child process exited with code ' + code)
                //this.console.text = "child process exited with code " + code;
    
                //this.addConsoleOutputToBuffer('child process exited with code' + code.toString());
                consoleOutput.push('child process exited with code' + code.toString());
    
                // Reload logs, which will likely be updated from the completed run
                //this.refreshLogStateSpace();
    
                res.status(200).send()
    
            })
    
    
    
            // Step 4: Display IML console interpreter results
    
            // Step 5: Display IML output
    
        }
        else{
            console.log(`ERROR! Cannot run pattern "${patternName}". No IML saved on file.`)
            res.status(500).send(`ERROR! Cannot run pattern "${patternName}". No IML saved on file.`)
        }
    }
    else{
        console.log(`ERROR! Address: "${path}" does not point to valid copy of worldtree.\nNOTE: If this is happening, then we are runing before save finishes executing.`)
        res.status(500).send(`ERROR! Address: "${path}" does not point to valid copy of worldtree.\nNOTE: If this is happening, then we are runing before save finishes executing.`)
    }
})


/*
 * Utility Functions
 */

// Handles File IO required to save a given pattern asset 
// (e.g. edge table, IML, etc..)
// TODO: want to make savePattern() easier to read, but not sure how to write this abstract function without recursion
function savePatternAsset(){

}

function safeExecSync(cmd) {
    try {
        var result1 = execSync(cmd);
    } catch (e) {
        console.log("execSync: ERROR: " + e);
        return;
    }
}

// Generates a date string
function getDate() {
    let today = new Date();

    let sec = String(today.getSeconds()).padStart(2, '0')
    let min = String(today.getMinutes()).padStart(2, '0')
    let hr = String(today.getHours()).padStart(2, '0')
    let dd = String(today.getDate()).padStart(2, '0');
    let mm = String(today.getMonth() + 1).padStart(2, '0'); //January is 0!
    let yyyy = today.getFullYear();

    today = `${sec}-${min}-${hr}-${mm}-${dd}-${yyyy}`;

    return today
}

// JSON stringify but pretty
function JSONStringify(obj) {
    return JSON.stringify(obj, null, 2)
}

// Convert tsv string to Dataframe
// {
//     field0: [0, 1, 2, ...],
//     field1: [0, 1, 2, ...],
//     field2: [0, 1, 2, ...],
// }
// complex header has notes and pattern markings preceed the data 
function parseTSV(data, delimiter, withComplexHeader) {

    var df = {}

    let lines = data.split('\n')

    let firstLine = 0   // Complex header takes up the first two lines

    // Extra bits loaded for complex headers. these have notes and ratings on the first two lines respectively
    if (withComplexHeader) {
        firstLine = 2
        df['notes'] = lines[0]

        let flags = lines[1].split('\t')
        df['isDone'] = parseBoolean(flags[0])
        df['isGood'] = parseBoolean(flags[1])
        df['isUncertain'] = parseBoolean(flags[2])
        df['isBad'] = parseBoolean(flags[3])
        df['isRedundant'] = parseBoolean(flags[4])
    }

    // Initialize the header
    let header = lines[firstLine].split(delimiter)
    for (let col of header) {
        df[col] = []
    }

    // Metadata stuff
    df['length'] = lines.length - 1 - firstLine // dont count first two lines with complex
    df['width'] = header.length
    df['header'] = header

    // push the data onto each head
    for (let i = firstLine + 1; i < lines.length; i++) {
        let line = lines[i]

        // Read column info
        let cols = line.split("\t")
        for (let j = 0; j < cols.length; j++) {
            let col = cols[j]
            let head = header[j]

            if (head == "hintRowUUIDs") {
                if (col.length == 0) {
                    df[head].push([]);
                } else {
                    df[head].push(col.split(","))           // Array
                }
            } else if (head == "hintWords") {
                if (col.length == 0) {
                    df[head].push([]);
                } else {
                    df[head].push(col.split(","))           // Array
                }
            } else if (head == "OPTIONAL") {
                df[head].push(parseBoolean(col));
            } else {
                df[head].push(col)
            }
        }
    }

    return df
}

// Convert tsv string to Dataframe
// More comperable to a pandas dataframe
function parseTSV2(data, delimiter, namedHeader = true, namedIndices = true) {

    var df = {}

    let lines = data.split('\n')

    // Read first line
    let firstLine = lines[0].split(delimiter)

    // Populate header
    let header = []
    // named header
    if (namedHeader) {
        for (let j = (namedIndices) ? 1 : 0; j < firstLine.length; j++) {
            let head = firstLine[j]
            header.push(head)
            df[head] = {}
        }
    }
    // unnamed header: use 0, 1, 2, ...
    else {
        for (let j = (namedIndices) ? 1 : 0; j < firstLine.length; j++) {
            df[j] = {}
            header.push(j)
        }
    }

    // Metadata stuff
    df['length'] = lines.length - 1
    df['width'] = header.length
    df['header'] = header
    df['isDone'] = parseBoolean(firstLine[0])  // a little hacky but here's my isDone flag for the tables
    let index = []

    // console.log(header)
    // console.log(df)

    // push the data onto each head
    // skip first line if it was a named header
    for (let i = (namedHeader) ? 1 : 0; i < lines.length; i++) {
        let line = lines[i]

        // Read column info
        let cols = line.split("\t")

        if (namedIndices) index.push(cols[0])
        else {
            if (namedHeader) index.push(i - 1)
            else index.push(i)
        }
        for (let j = (namedIndices) ? 1 : 0; j < cols.length; j++) {
            let col = cols[j]
            let head = (namedIndices) ? header[j - 1] : header[j]

            if (namedIndices) df[head][cols[0]] = col
            else {
                if (namedHeader) df[head][i - 1] = col
                else df[head][i] = col
            }
        }
    }

    df['index'] = index

    return df
}

// Convert tsv string to Dataframe
// {
//     field0: [0, 1, 2, ...],
//     field1: [0, 1, 2, ...],
//     field2: [0, 1, 2, ...],
// }
function stringifyDF(df, delimiter, withComplexHeader) {

    var data = ""

    // Complex header has a notes field and flag list
    if (withComplexHeader) {
        data += df['notes'] + '\n'
        data += `${stringifyBoolean(df['isDone'])}\t${stringifyBoolean(df['isGood'])}\t${stringifyBoolean(df['isUncertain'])}\t${stringifyBoolean(df['isBad'])}\t${stringifyBoolean(df['isRedundant'])}\n`
    }

    // The header
    let header = df.header
    for (let i = 0; i < header.length; i++) {
        let head = header[i]
        if (i !== header.length - 1) {
            data += `${head}${delimiter}`
        }
        else {
            data += `${head}\n`
        }
    }

    // The rows
    for (let i = 0; i < df.length; i++) {

        // For each column
        for (let j = 0; j < header.length; j++) {
            let head = header[j]
            let col = df[head][i]
            if (j !== header.length - 1) {
                data += `${col}${delimiter}`
            }
            else {
                data += `${col}\n`
            }
        }
    }

    // Snip off the last newline
    data = data.substring(0, data.length - 1)

    return data
}

// stringify for the second kind of dataframe
function stringifyDF2(df, delimiter = '\t', namedHeader = true, namedIndices = true) {

    var data = ""

    let header = df.header
    let indices = df.index

    // The header
    if (namedHeader) {

        // empty space in (0,0)
        if (namedIndices) data += (df['isDone']) ? 'true\t' : 'false\t'

        for (let i = 0; i < header.length; i++) {
            let head = header[i]
            data += `${head}${delimiter}`
        }

        // Newline at end
        data = data.substring(0, data.length - 1) + '\n'
    }

    // The rows
    for (let i = 0; i < df.length; i++) {

        if (namedIndices) data += `${indices[i]}\t`

        // For each column
        for (let j = 0; j < header.length; j++) {
            let head = header[j]
            let idx = indices[i]

            let col = df[head][idx]
            data += `${col}${delimiter}`
        }

        // Newline at end
        data = data.substring(0, data.length - 1) + '\n'
    }

    // Snip off the last newline
    data = data.substring(0, data.length - 1)

    return data
}

// Removes the '.xxx' from the end
// Hard coded extension length
function getFileName(file) {
    return file.substring(0, file.length - 4)
}

// Get a patterns index based on its name
function getPatternIdx(patternName) {
    var patternIdx = -1
    for (let i = 0; i < possiblePatterns.length; i++) {
        if (possiblePatterns[i] === patternName) {
            patternIdx = i
        }
    }
    return patternIdx
}

// Get dataset index based on its name
function getDatasetIdx(dataset) {
    var datasetIdx = -1
    for (let i = 0; i < possibleDatasets.length; i++) {
        if (possibleDatasets[i] === dataset) {
            datasetIdx = i
        }
    }
    return datasetIdx
}

// Change the possible patterns based onn the current label
function getPatternSelections(data_addr) {

    // read in possible files for the current label and refresh possible patterns
    let possiblePatternFiles = getFiles(data_addr)

    // Sort alphabetically
    possiblePatternFiles.sort()

    // clean filenames for user selections
    let possiblePatterns = []
    for (let i = 0; i < possiblePatternFiles.length; i++) {
        possiblePatterns.push(getFileName(possiblePatternFiles[i]))
    }

    return possiblePatterns
}

/*
 * File I/O Helpers
 */

// From https://stackoverflow.com/questions/18112204/get-all-directories-within-directory-nodejs

// Returns all directories in a file path
function getDirectories(path) {
    return fs.readdirSync(path).filter(function (file) {
        return fs.statSync(path + '/' + file).isDirectory();
    });
}

// Returns all .tsv files in a file path
function getFiles(path) {
    return fs.readdirSync(path).filter(function (file) {
        // check that it is a file of valid format
        return fs.statSync(path + '/' + file).isFile() && 'tsv' === file.substring(file.lastIndexOf(".") + 1);
    });
}

// Determines available directories in the paterns/ folder
function getDataSets() {

    // Read in the possible labels from the FS
    var folders = getDirectories(PATTERNS_PATH)
    for (let i = 0; i < folders.length; i++) {
        let folder = folders[i]
        var filesInFolder = getFiles(PATTERNS_PATH + "/" + folder)
        possibleDatasetDirs.push({ label: folder, files: filesInFolder })
    }

    // Interperet labels from the dir structures
    for (let i = 0; i < possibleDatasetDirs.length; i++) {
        possibleDatasets.push(possibleDatasetDirs[i].label)
    }
}

// 
function initialize() {
    if (tablestoreGoogleSheets.checkIfRowsLoaded()) {
        rows = tablestoreGoogleSheets.getLoadedRows();
        console.log("Server: Rows returned: " + rows.length)

        // Load questions and explanations
        questions = tablestore.loadQuestions(`${QUESTIONS_PATH}/ARC-QC+EXPL-Train-1.tsv`, rows);

        // Read in and interperet the labels and grids available        
        return true;

    } else {
        setTimeout(initialize, 1000)
        return false;
    }
}

/*
 * File I/O
 */

function getPrefs() {

    let data = {}

    try {
        let fileData = fs.readFileSync('prefs.ini', 'utf-8')
        let lines = fileData.split('\n')
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i]

            let firstChar = line[0]
            if (firstChar !== ';' && firstChar !== '[') {
                let keyValPair = line.split('=')
                if (keyValPair.length >= 2) {
                    let key = keyValPair[0].trim()
                    let val = keyValPair[1].trim()
                    if(val == 'true') val = true
                    if(val == 'false') val = false

                    data[key] = val
                }
            }
        }

    } catch (err) {
        console.log('\tWarning! No prefs.ini file found. Will be using default parameters.')
    }
    return data
}

// Load IML from file if it exists
function loadIML(patternName, dataset){
    // console.log(`loadIML(${patternName}, ${dataset})`)

    let imlStr = ""

    // Determine if there is a manual IML on file, if not switch to the auotmatically generated file
    var fileName =                          `${PATTERNS_PATH}${dataset}/IML_manual/${patternName}.iml`
    if(!fs.existsSync(fileName)) fileName = `${PATTERNS_PATH}${dataset}/IML_auto/${patternName}.iml`

    // If there is an IML copy on file load it: otherwise leave the string empty
    if(fs.existsSync(fileName)){
        imlStr = fs.readFileSync(fileName, 'utf8')
    }

    return imlStr
}

function loadPatterns(dataset) {

    // Load the pattern seelctions
    let data_addr = `${PATTERNS_PATH}/${dataset}/`
    let patternSelections = getPatternSelections(data_addr)

    // Initialize data obj, (this is what we send to the client)
    let data = {
        patterns: [],
        patternNames: patternSelections,
    }

    // Loop through each patterns selection and load them into this array
    let patterns = []
    for (let i = 0; i < patternSelections.length; i++) {
        let patternName = patternSelections[i]

        let pattern = loadPattern(patternName, dataset)
        pattern.userIMLEdits = {}

        patterns.push(pattern)
    }

    // Package the patterns and return the package
    data.patterns = patterns
    return data
}

function loadPattern(patternName, dataset) {
    let data_addr = `${PATTERNS_PATH}/${dataset}/`
    let filename = `${data_addr}${patternName}.tsv`

    let pattern = {}

    let data = fs.readFileSync(filename, 'utf8')
    if (typeof data === 'string') {
        
        pattern = parseTSV(data, "\t", true)

        pattern.iml = loadIML(patternName, dataset)

        if(fs.existsSync(`${PATTERNS_PATH}${dataset}/IML_manual/${patternName}.iml`)) pattern.isManualIML = true
        else pattern.isManualIML = false

    }
    else {
        console.log(`Error! Pattern file present but corrupted. type: "${typeof data}"`)
    }

    return pattern
}

// Load JSON of IML edits
function loadUserIMLEdits(filename){
    
    let user_edits = {}

    if(fs.existsSync(filename)){
        let fileStr = fs.readFileSync(filename, 'utf8')
        user_edits = JSON.parse(fileStr)
    }
    
    return user_edits
}

function loadEdgeTables(dataset) {

    // Load the pattern seelctions
    let data_addr = `${PATTERNS_PATH}/${dataset}/edge_tables/`
    let edgeTableNames = getPatternSelections(data_addr)


    // Initialize data obj, (this is what we send to the client)
    let data = {
        edgeTables: [],
        edgeTableNames: edgeTableNames,
    }

    // Loop through each patterns selection and load them into this array
    let edgeTables = []
    for (let i = 0; i < edgeTableNames.length; i++) {
        let edgeTableName = edgeTableNames[i]

        let edgeTable = loadEdgeTable(data_addr + `${edgeTableName}.tsv`)

        edgeTables.push(edgeTable)
    }

    // Package the patterns and return the package
    data.edgeTables = edgeTables

    return data
}

function loadEdgeTablesMap(dataset) {

    // Load the pattern seelctions
    let data_addr = `${PATTERNS_PATH}/${dataset}/edge_tables/`

    // Get edge table names
    let edgeTableNames = getPatternSelections(data_addr)

    // Initialize data obj, (this is what we send to the client)
    let edgeTables = {}

    // Loop through each patterns selection and hash them into a map by name
    for (let i = 0; i < edgeTableNames.length; i++) {
        let edgeTableName = edgeTableNames[i]

        let edgeTable = loadEdgeTable(data_addr + `${edgeTableName}.tsv`)

        // cut off 'edges_' when hashing
        edgeTables[edgeTableName.substring(6)] = edgeTable
    }

    return edgeTables
}

function loadEdgeTable(filename) {

    let edgeTable = {}

    let data = fs.readFileSync(filename, 'utf8')
    if (typeof data === 'string') {
        edgeTable = parseTSV2(data, "\t")

        // console.log("----------------------------------------------------------------------------------------")
        // console.log(edgeTable)
    }
    else {
        console.log(`Error! Pattern file present but corrupted. type: "${typeof data}"`)
    }

    return edgeTable
}

function loadRowRoleFreqs() {

    fs.readFile(ROW_ROLE_FREQS_PATH, function (err, data) {
        if (err) {
            console.log(`\tERROR! Missing rowRoleFreqs file. Expecting it to be at: "${ROW_ROLE_FREQS_PATH}".`)
        }
        else {
            if (typeof data !== 'undefined') {
                rowRoleFreqs = JSON.parse(data)
                console.log("RowRoleFreqs file loaded.")
            }
            else {
                console.log("Error! RowRoleFreqs file present but corrupted.")
            }
        }
    })
}

// Parse text 'true' as a boolean true, else false
function parseBoolean(strIn) {
    if (strIn) {
        if (strIn.toLowerCase() == 'true') return true;
    }
    // Default return
    return false;
}

function stringifyBoolean(boolIn) {
    if (boolIn) return 'true'
    else return 'false'
}

// Pulls in the lemmatizer file and created a hashmap
function loadLemmatizer() {
    let lemmaFile = fs.readFileSync(LEMMA_PATH, 'utf8')
    // NOTE THERE ARE \r in this file
    var lines = lemmaFile.split("\r\n");

    // read through eachline of the file
    // File is formatted like '<lemma> <word>'
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].split('\t')
        let lemma = line[0]
        let word = line[1]

        lemmatizer[word] = lemma

        //console.log(`${lemma} -- ${word}`)
    }

    //console.log(lemmatizer)

    lemmatizerReady = true
}
