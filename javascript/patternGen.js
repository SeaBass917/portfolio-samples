///////////////////////////////////////////////////////////////////////////////////////////////////
// Authors:         Sebastian Thiem, Peter Jansen
//
// Sponsor:         
// 
// Project Name:    Syncronicity
// File Name:		patternGen.js
// Create Date:     14 Sept 2019
//
// Description:     Handles all functional elements of the pattern annotation tool: Syncronicity.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Globals (Forgive me Gregg):
 */

var POLLING_INTERVAL =  128 //ms
var PING_CYLES =        32  //clk cycles
var PING_TIMER =        32  // the timer itself
var IML_VIEWER_REFRESH_CYCLES = 16  // Trigger a refresh on the IML viewer if it is open
var IML_VIEWER_REFRESH_TIMER = 16  // the timer itself
var PUBLISH_COOLDOWN_CYCLES = 4 //clk cycles
var PUBLISH_COOLDOWN =  0       // the timer itself
var READY_TO_POLL = false
var AUTOSAVE = true

// IML code editor
var editorIML
require.config({ paths: { 'vs': 'monaco-editor/min/vs' }});

// Tablestore stuff
var tableRows = []
var tableRowsAnnotated = []
var tableMap = {}
var tableMapAnnotated = {}
var tableHeaders = {}
var questions = []
var rowRoleFreqs = {}
var tableNames = new Set()

// Stopword list
var stopWords = [
    "i", "me", "my", "myself", "we", "our", 
    "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", 
    "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", 
    "whom", "this", "these", "those", "am", 
    "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", 
    "doing", "a", "an", "the", "and", "but", "if", "or", 
    "because", "as", "until", "while", "of", "at", 
    "about", "against", "between", "then", 
    "once", "here", "there", "when", "where", "why", "how", 
    "all", "any", "both", "each", "few", "more", "most", 
    "other", "some", "such", "no", "nor", "not", "only", 
    "own", "same", "so", "very", "s", "t", 
    "just", "dont", "now", "from", "", 
]

var stopWords_special = [
    "energy", "probability", "either", "reaction",
    "as", "at", "by", "in", "for", "on", "than", "to", "with",
    "too", "often", "usually", "can", "will", "should", "that",
]

// Preference sent from the server
var prefs = null

// Timer for the publish button so that we dont spam

// Lemmatizer
var lemmatizer = {}

// Pattern variables
var patterns = []
var patternNames = []
var patternIdx = -1
var pattern = null

// Edge tables that store the edges for each completed pattern
var edgeTablesMap = {}

var currentDataset =  ""
var currentPattern =  ""
var currentRow = "row_0"

// Row ratings
var RATING_CENTRAL      =   "CENTRAL"
var RATING_CENTRALSW    =   "CENTRALSW"
var RATING_GROUNDING    =   "GROUNDING"
var RATING_LEXGLUE      =   "LEXGLUE"
var RATING_MAYBE        =   "MAYBE"
var RATING_BAD          =   "BAD"
var RATING_UNRATED      =   "UNRATED"

// Constraint Classes
var CLASS_UNRATED                   = 0
var CLASS_STOPWORD                  = 1
var CLASS_LEXICALIZED               = 2
var CLASS_POSSIBLEVARIABLIZED       = 3
var CLASS_VARIABLIZED               = 4
var CLASS_VARIABLIZED_LEXICALIZED   = 5
var CLASS_POS            = 6
var constraintClassList = [
    // CLASS_UNRATED,
    CLASS_STOPWORD,
    CLASS_LEXICALIZED,
    // CLASS_POSSIBLEVARIABLIZED,
    CLASS_VARIABLIZED,
    CLASS_VARIABLIZED_LEXICALIZED,
    CLASS_POS,
]

// Warning messages
var MISSING_HINTROW_WARNING = '<br><font color="#d94801"><i>Warning: Swappable row requires hint rows</i></font>'
var MISMATCHED_TABLES_WARNING = '<br><font color="#d94801"><i>Warning: Hint rows must come from the same table</i></font>'
var FAILING_CONSTRAINTS_WARNING = '<br><font color="#d94801"><i>Warning: One or more hints rows present do not meet the constraints applied to this row</i></font>'
var POSSIBLE_MISPLACED_HINTROW_WARNING = '<br><font color="#D66C4F"><i>Warning: Possibly misplaced hintrow</i></font>'

// String constants
var OPTIONAL_ROW_STR        =   "<br><font color=\"grey\"><i>Optional</i></font>" 

// Flags for filtering the presented patterns
var showDone = true
var showGood = true
var showUncertain = false
var showBad = false
var showRedundant = false
var showUnmarked = true

// List of current ratings
var rowRatingsForCurrentPattern = []

// Flags
var inATextBox = false
var changesMadeSinceLastSave = false
var changesSinceLastIMLPopulate = true
var dataInEditor = false

// stores the last table for easy jump back to
var lastTableName = ""

// Global for pattern Query box
var patternQuery = ''
var preQueryPattern = ''

// Stack of pattern selection History
var patternSelectionStack = []

// Popup windows
let rqWindow
let ovWindow

// Cytoscape object
var cy

// the stylesheet for the graph
var cyStyle = [
    {
        selector: 'node',
        style: {  //["#eff3ff","#c6dbef","#9ecae1","#6baed6","#3182bd","#08519c"]
            'background-color': '#c6dbef',
            'border-color': '#6baed6',
            'border-style': 'solid',
            'border-width': '1px',
            'font-size': 12,
            'height': 'label',
            'label': 'data(label)',             // What text to draw on the nodes
            'opacity': 0.9,
            'padding': '5px',
            'text-halign':'center',
            'text-valign':'center',
            'text-wrap':'wrap',                 // Allow multi-line (wrapped) node text (using \n markers)
            'width': 'label',
            'shape': 'round-rectangle',
        },
    },
    {
        selector: 'node.G',
        style: {  //["#fff5eb","#fee6ce","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#a63603","#7f2704"]
            'background-color': '#fdae6b',
            'border-color': '#f16913',
        },
    },
    {
        selector: 'node.SW',
        style: {  //["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#08519c","#08306b"]
            'background-color': '#9ecae1',
            'border-color': '#2171b5',
        },
    },
    {
        selector: 'node.C',
        style: {  //["#f7fcf5","#e5f5e0","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#006d2c","#00441b"]
            'background-color': '#a1d99b',
            'border-color': '#238b45',
        },
    },
    {
        selector: 'node.LG',
        style: {
            'background-color': '#CAB2D6',
            'border-color': '#B685D1',
        },
    },
    {
        selector: 'node.warning',
        style: {
            'border-color': '#D66C4F',
            'border-width': '6px',
        },
    },
    {
        selector: 'node.problem',
        style: {
            'border-color': 'red',
            'border-width': '6px',
        },
    },
    {
        selector: 'node:selected',
        style: {
            'border-color': '#08519c',
            'border-width': '3px',
        },
    },
    {
        selector: 'edge.autorotate',     
        style: {
            'color': '#000000',
            'font-size': 8,
            'label': 'data(label)',
            'line-color': '#6baed6',
            'opacity': 1.0,
            'text-wrap':'wrap',
            'width': 1,
            'curve-style': 'bezier',
        },
    },
    {
        selector: 'edge.soft',     
        style: {
            'color': '#a0a0a0',
            'label': 'data(label)',
            'line-color': '#c0c0c0',
            'opacity': 0.85,
            'text-wrap':'wrap',
            'width': 1,
            'curve-style': 'bezier',
        },
    },
    {
        selector: 'edge.autorotate:selected',     
        style: {
            'color': 'red',
            'label': 'data(label)',
            'line-color': '#6E8F9C',
            'text-wrap':'wrap',
        },
    },
    {
        selector: 'edge.problem',     
        style: {
            'line-color': 'red',
            'width': '3px',
        },
    },
]

/*
 * Utility Functions
 */

function isContentTag(tag){
    if(tag.length >= 2){
        let firstTag = tag.substring(0, 2)
        if(["NN", "VB", "JJ", "RB", "IN", "CD"].includes(firstTag)) return true
    }
    return false
}

// Syncronous print message
function print(message){
    console.log(deepCopy(message))
}

// Compare element by element
function arrayDeepCompare(a, b){
    let isEqual = true
    if(a.length == b.length){
        for(let i in a){
            if(a[i] != b[i]) isEqual = false
        }
    }
    else{
        isEqual = false
    }
    return isEqual
}

// Loops through the 2d matrix and removes rows taht are duplicates of other rows
function removeDuplicateRows(matrix){

    let filteredMatrix = []
    for(let row of matrix){

        let rowIsUnique = true
        for(let row_i of filteredMatrix){
            if(arrayDeepCompare(row, row_i)) rowIsUnique = false
        }

        if(rowIsUnique) filteredMatrix.push(row)
    }

    return filteredMatrix
}

// Simple duplicate shredder, wont work with lists 
function removeDuplicatesFromListQuick(list){
    return list.filter((a, i) => list.indexOf(a) === i)
}

function isNumber(str){
    if(typeof str == 'string') return str >= 0 || 0 < str
    else return false
}

function classEnumToString(classes){
    let str = ""
    switch(classes){
        case CLASS_UNRATED:
            str = "UNK"
            break
        case CLASS_STOPWORD:
            str = "STOP"
            break
        case CLASS_LEXICALIZED:
            str = "LEX"
            break
        case CLASS_POSSIBLEVARIABLIZED:
            str = "POSVAR"
            break
        case CLASS_VARIABLIZED:
            str = "VAR"
            break
        case CLASS_VARIABLIZED_LEXICALIZED:
            str = "VAR & LEX"
            break
    }
    return str
}

// Allowing hashing of any string
String.prototype.hashCode = function() {
    var hash = 1;
    if (this.length == 0) {
        return hash;
    }
    for (var i = 0; i < this.length; i++) {
        var char = this.charCodeAt(i);
        hash = ((hash<<5)-hash)+char;
        hash = hash & hash; // Convert to 32bit integer
        hash = (hash < 0)? (-1) * hash : hash
        hash = (hash)? hash : 1
    }
    return hash
}

// Remove last ", " from a string
function trimComma(s){
    let len = s.length
    if(len > 0) s = s.substring(0, len-2)
    return s
}

// Set union
function union(a, b){
    return new Set([...a, ...b])
}

// Set intersection
function intersection(a, b){
    return new Set([...a].filter(i => b.has(i)))
}

// Send a message to the server asking for the data to be published to git
function publishChanges(){

    // Safety cooldown so that we dont spam github
    if(PUBLISH_COOLDOWN == 0 ){
        PUBLISH_COOLDOWN = PUBLISH_COOLDOWN_CYCLES
        stylePublishingButton()

        $.post('/PostPublish', {}, function(res, status){
            if(status === 'success'){
                let message = res.message
                let update = res.update
                let mergeConflicts = res.mergeConflicts

                console.log(message)
                console.log(update)
                console.log(mergeConflicts)

                // Check for merge conflicts
                if(!mergeConflicts){
                    
                    // Changes were made externally, read in the new patterns
                    if(update !== null){
                        console.log('Recieved update from server. Reloading patterns.')
    
                        let patternsNew = update.patterns
                        let patternNamesNew = update.patternNames
    
                        // Save what pattern we are currently on
                        let currentPattern = patternNames[patternIdx]
    
                        // Find the new patternIdx
                        patternIdx = patternNamesNew.indexOf(currentPattern)
                        
                        // If we come accross an exceptions (e.g. the current pattern is miissing now)
                        // We need to warn the user
                        try{
                            if(patternIdx >= 0){
        
                                patternNames = patternNamesNew
                                populatePatternSeletor()
                                selectByValue('patternSelector', currentPattern)
        
                                // Update the patterns list
                                patterns = patternsNew
        
                                // NOTE: This wont work if their are discrepencies in how the data is stored on the server vs writen to locally
                                // (e.g. locally stored as 1 but on the server its "1")
                                // For this reason it is not worth it to check for changes to this page. Especially since all other patterns edited in this session are also at risk anyway

                                // Check to see if the pattern we have been working on is still the same on the server
                                // we aren't checking all patterns edited this session, just the one we're currently viewing
                                // if(JSON.stringify(pattern) !== JSON.stringify(patterns[patternIdx])){
                                //     console.log(pattern)
                                //     console.log(patterns[patternIdx])
                                //     throw new Error('ERROR! Pattern currently working on was edited externally!')
                                // }
                            } else{
                                throw new Error('ERROR! Pattern currently working on was renamed externally!')
                            }
                        }
                        catch(err){
                            let message = err.message
                            document.body.style.backgroundColor = 'black'
                            document.getElementById('toolName').innerHTML = message + ' Please refresh the page.'
                            document.getElementById('toolName').style.color = 'red'
                        }
                    }
                }
                else{
                    document.body.style.backgroundColor = 'black'
                    document.getElementById('toolName').innerHTML = 'ERROR! MERGE CONFLICTS ON AUTO-PULL. Please consult the git status.'
                    document.getElementById('toolName').style.color = 'red'
                }

                console.log(message)
            }
            else{
                console.log(res)
            }
        })
    }
}

function parseBoolean(str){
    let strLower = str.toLowerCase()
    if(strLower === 'true') return true
    else return false
}

// Change a specified selectors selection to the given value
function selectByValue(selectorId, val){
    
    let selectionToDeSelect = $(`#${selectorId} option:selected`)[0]
    let selectionToSelect = $(`#${selectorId} option[value="${val}"]`)[0]

    // Guard code, if we call with a bad value then let the console know
    if(typeof selectionToSelect !== 'undefined'){
        selectionToDeSelect.selected = false
        selectionToSelect.selected = true
        
        return true
    } else{
        console.log(`%cWarning! Cannot set selector "${selectorId}" to value "${val}".`, 'color:darkorange')
        return false
    }
}

// Determine if there exists levical overlap between the two lists of words
function cellOverlap(words_i, words_j){

    let lexicalOverlap = false
    for(word_i of words_i){
        let w_i = lemmatize(word_i)
        if(!stopWords.includes(w_i)){   // No stop word matches
        
            for(word_j of words_j){
                let w_j = lemmatize(word_j)
                if(!stopWords.includes(w_j)){   // No stop word matches
                    
                    if(w_i === w_j) {
                        lexicalOverlap = true
                    }
                }
            }
        }
    }

    return lexicalOverlap
}
// Lemmatize the given word
function lemmatize(word){
    let word_lower = word.toLowerCase()
    let lemma = lemmatizer[word_lower]
    if(typeof lemma === 'undefined') lemma = word_lower // lemmas and things not in the dict will return undefined
    return lemma
}

// "a | dog | | | is | a kind of | animal | (KINDOF UID:0000-0000-0000-0000)" -> "a dog is a kind of animal"
function cleanTextDel(textDel){

    let cols = textDel.split(" | ")
    cols.pop()  // remove last col
    let text = ``
    for(col of cols){
        if(col.length > 0) text+=col+' '
    }

    return text
}

function getObjKeys(obj){
    return Object.keys(obj).filter(function(key) {return obj.hasOwnProperty(key)})
}

// Handles attatching the tooltip api to elements that use it
function attatchToolTips(){

    // The buttons that have tooltips
    let buttons = document.getElementsByTagName('button')
    for(button of buttons){
        let text = button.getAttribute('data-tooltip')
        if(text){
            new Tooltip(button, {
                title: text,
                trigger: "hover",
                placement: "bottom",
                // delay: { // doesnt work, can cause the tooltip to stay active
                //     show: 300, 
                //     hide: 100 
                // },
            });
        }
    }

    // Icons that may have tooltips
    let eles = document.getElementsByTagName('i')
    for(ele of eles){
        let text = ele.getAttribute('data-tooltip')
        if(text){
            new Tooltip(ele, {
                title: text,
                trigger: "hover",
                placement: "bottom",
                // delay: { // doesnt work, can cause the tooltip to stay active
                //     show: 300, 
                //     hide: 100 
                // },
            });
        }
    }
}

// Empty the lists in the pattern for the frefresh
function clearPattern() {    
    pattern["BACKGROUND_count"] = []
    pattern["CENTRAL_count"] = []
    pattern["GROUNDING_count"] = []
    pattern["LEXGLUE_count"] = []
    pattern["ROLE_count"] = []
    pattern["RATING"] = []
    pattern["ROW"] = []
    pattern["TABLE"] = []
    pattern["UID"] = []
    pattern["length"] = 0

    pattern["lastCount"] = []
    pattern["hintRowUUIDs"] = []
    pattern["hintWords"] = []    
    pattern["rowNotes"] = []    
    pattern["OPTIONAL"] = []    
}

// Increment current row variable and checks for bounds
// Saturating counter
function incrementCurrentRow(){

    let next = document.getElementById(currentRow).nextElementSibling

    // Check for bounds
    if(next !== null){
        currentRow = next.id
        focusRowSelected()
    }
}

// Increment current row variable and checks for bounds
// Saturating counter
function decrementCurrentRow(){

    let prev = document.getElementById(currentRow).previousElementSibling

    // Check for bounds
    if(prev !== null){
        currentRow = prev.id
        focusRowSelected()
    }
}

// 
function deepCopy(obj){
    return JSON.parse(JSON.stringify(obj))
}

// 
function findHTMLRowRating(row){
    let rowIdx = parseInt(row.id.substring(4))
    return pattern['RATING'][rowIdx]
}

// Get the selected option for the given selector id
function getSelectedOptionValue(id){
    return $(`#${id} option:selected`).val()
}

// Gets the number of times row is used as central and grounding
function getCountForRow(uid){
    let query = querySeedRows([uid])

    return query.get(uid)
}

// 
function mkSeedElement(row, i){

    let uid = pattern['UID'][i]    
    let count  = getCountForRow(uid)
    return `<li>(${count}) -- ${row}</li>\n`
}

// Get the idx of a pattern based on the name
function getPatternIdx(name){
    let idx = -1
    for(let i = 0; i < patternNames.length; i++){
        let patternName = patternNames[i]

        if(patternName === name){
            idx = i
        }
    }
    return idx
}

// Flag the pattern for a given state of completeness
function markPattern(flagName){
    
    // (r) ${patternName}
    // (b) ${patternName}
    // (?) ${patternName}
    // (g) ${patternName}
    // (d) ${patternName}
    //     ${patternName}
    let patternName = patternNames[patternIdx]
    let selectorOption = null
    switch(flagName){
        case 'DONE':
            if(pattern['isDone']){
                pattern['isDone'] = false
            }
            else{
                pattern['isDone'] = true
                pattern['isGood'] = false
                pattern['isUncertain'] = false
                pattern['isBad'] = false
                pattern['isRedundant'] = false
            }

            // Update the selector
            selectorOption = $(`#patternSelector option[value="${patternName}"]`)[0]
            selectorOption.innerHTML = `(d) ${patternName}`
            break;
        
        case 'GOOD':
            if(pattern['isGood']){
                pattern['isGood'] = false
            }
            else{
                pattern['isDone'] = false
                pattern['isGood'] = true
                pattern['isUncertain'] = false
                pattern['isBad'] = false
                pattern['isRedundant'] = false
            }

            // Update the selector
            selectorOption = $(`#patternSelector option[value="${patternName}"]`)[0]
            selectorOption.innerHTML = `(g) ${patternName}`
            break;
        
        case 'UNCERTAIN':   // Let uncertain be independant
            if(pattern['isUncertain']){
                pattern['isUncertain'] = false
            }
            else{
                pattern['isUncertain'] = true
            }

            // Update the selector
            selectorOption = $(`#patternSelector option[value="${patternName}"]`)[0]
            selectorOption.innerHTML = `(?) ${patternName}`
            break;

        case 'BAD':
            if(pattern['isBad']){
                pattern['isBad'] = false
            }
            else{
                pattern['isDone'] = false
                pattern['isGood'] = false
                pattern['isUncertain'] = false
                pattern['isBad'] = true
                pattern['isRedundant'] = false
            }

            // Update the selector
            selectorOption = $(`#patternSelector option[value="${patternName}"]`)[0]
            selectorOption.innerHTML = `(b) ${patternName}`
            break;
        
        case 'REDUNDANT':
            if(pattern['isRedundant']){
                pattern['isRedundant'] = false
            }
            else{
                pattern['isDone'] = false
                pattern['isGood'] = false
                pattern['isBad'] = false
                pattern['isRedundant'] = true
            }

            // Update the selector
            selectorOption = $(`#patternSelector option[value="${patternName}"]`)[0]
            selectorOption.innerHTML = `(r) ${patternName}`
            break;
        
        default:
            pattern['isDone'] = false
            pattern['isGood'] = false
            pattern['isUncertain'] = false
            pattern['isBad'] = false
            pattern['isRedundant'] = false

            // Update the selector
            selectorOption = $(`#patternSelector option[value="${patternName}"]`)[0]
            selectorOption.innerHTML = `${patternName}`
    }

    styleFlagButtons()

    changesMadeSinceLastSave = true
}

// Uses classes to switch the style of selections beteen on or not on, some would even dare to say 'off'
function styleFlagButtons(){
    if(pattern['isDone']) document.getElementById('doneButton').classList.add('on')
    else document.getElementById('doneButton').classList.remove('on')
    if(pattern['isGood']) document.getElementById('goodButton').classList.add('on')
    else document.getElementById('goodButton').classList.remove('on')
    if(pattern['isUncertain']) document.getElementById('uncertainButton').classList.add('on')
    else document.getElementById('uncertainButton').classList.remove('on')
    if(pattern['isBad']) document.getElementById('badButton').classList.add('on')
    else document.getElementById('badButton').classList.remove('on')
    if(pattern['isRedundant']) document.getElementById('redundantButton').classList.add('on')
    else document.getElementById('redundantButton').classList.remove('on')
}

// Uses classes to switch the style of selections beteen on or not on, some would even dare to say 'off'
function styleFilterButtons(){
    if(showDone) document.getElementById('filterDoneButton').style.color = '#000000'
    else document.getElementById('filterDoneButton').style.color = '#d1d1d1'
    if(showGood) document.getElementById('filterGoodButton').style.color = '#000000'
    else document.getElementById('filterGoodButton').style.color = '#d1d1d1'
    if(showUncertain) document.getElementById('filterUncertainButton').style.color = '#000000'
    else document.getElementById('filterUncertainButton').style.color = '#d1d1d1'
    if(showBad) document.getElementById('filterBadButton').style.color = '#000000'
    else document.getElementById('filterBadButton').style.color = '#d1d1d1'
    if(showRedundant) document.getElementById('filterRedundantButton').style.color = '#000000'
    else document.getElementById('filterRedundantButton').style.color = '#d1d1d1'
    if(showUnmarked) document.getElementById('filterUnmarkedButton').style.color = '#000000'
    else document.getElementById('filterUnmarkedButton').style.color = '#d1d1d1'

    // console.log(`
    //     if(${showDone}) document.getElementById('filterDoneButton').style.color = '#000000'
    //     else document.getElementById('filterDoneButton').style.color = '#d1d1d1'
    //     if(${showGood}) document.getElementById('filterGoodButton').style.color = '#000000'
    //     else document.getElementById('filterGoodButton').style.color = '#d1d1d1'
    //     if(${showUncertain}) document.getElementById('filterUncertainButton').style.color = '#000000'
    //     else document.getElementById('filterUncertainButton').style.color = '#d1d1d1'
    //     if(${showBad}) document.getElementById('filterBadButton').style.color = '#000000'
    //     else document.getElementById('filterBadButton').style.color = '#d1d1d1'
    //     if(${showRedundant}) document.getElementById('filterRedundantButton').style.color = '#000000'
    //     else document.getElementById('filterRedundantButton').style.color = '#d1d1d1'
    //     if(${showUnmarked}) document.getElementById('filterUnmarkedButton').style.color = '#000000'
    //     else document.getElementById('filterUnmarkedButton').style.color = '#d1d1d1'
    // `)
}

// Blurs and focuses the back prev and forward buttons
function styleMobilityButtons(){

    // Gray out the movement buttons
    let currOption = $('#patternSelector option:selected')[0]
    // console.log(currOption.previousElementSibling)
    if(currOption.previousElementSibling) $('#prevButton').css('color', 'black')
    else $('#prevButton').css('color', 'lightgrey')
    if(currOption.nextElementSibling) $('#nextButton').css('color', 'black')
    else $('#nextButton').css('color', 'lightgrey')
    
    if(patternSelectionStack.length > 1) $('#lastButton').css('color', 'black')
    else $('#lastButton').css('color', 'lightgrey')
}

// Blurs and focuses the publishing button when its on/off cooldown
function stylePublishingButton(){
    if(PUBLISH_COOLDOWN === 0) document.getElementById('publishButton').style.color = '#000000'
    else document.getElementById('publishButton').style.color = '#d1d1d1'
}

/*
 * Tablestore/Dataframes Utility
 */

// Ensures that text boxes have a flag that prevents macros while in use
function addFlaggerOnTextBoxes(){

    // Listener on textboxes to overwrite macros
    let textBoxes = $('input[type=text]')
    for(let i = 0; i < textBoxes.length; i++){
        let textBox = textBoxes[i]
        textBox.addEventListener('focus', function(){
            inATextBox = true
        })
        textBox.addEventListener('blur', function(){
            if(document.getElementById("imlVarView").style.display != 'flex') inATextBox = false
        })
    }
}

// Gte the index of the row in the pattern with the given UID
function getRowIdx(uid){
    for(let i = 0; i < pattern.length; i++){
        if(pattern['UID'][i] == uid) return i
    }
    return -1
}

function isSeedRow(rating){
    return rating == RATING_CENTRAL || rating == RATING_CENTRALSW || rating == RATING_GROUNDING
}

function isApprovedRow(rating){
    return rating == RATING_CENTRAL || rating == RATING_CENTRALSW || rating == RATING_GROUNDING || rating == RATING_LEXGLUE
}

function isSwappableRow(rating){
    return rating == RATING_CENTRALSW || rating == RATING_GROUNDING
}

// Find a tablestore row by uuid
function findRowByUUID(uuid) {
    for (var i=0; i<tableRows.length; i++) {            
        if (tableRows[i].uuid == uuid) {                
            return tableRows[i];
        }
    }
    return {
        text: `ERROR! No row in tablestore for uid: "${uuid}".`,
        tableName: "N/a",
        textDel: `(N/a, UID: ${uuid})`,
        uuid: uuid,
    }
}

// Returns the seed row UIDs from the current selected pattern
function getSeedRowUIDs(includeGrounding=false, includeHintRows=true, includeLexGlue=false) {
    var uuidsOut = []

    // Check that hintRows datastructure is populated (if using hint rows is enabled)
    if (includeHintRows == true) {
        if (!pattern.hasOwnProperty('hintRowUUIDs')) {                
            pattern['hintRowUUIDs'] = [];
            for (let a=0; a<pattern.length; a++) { pattern['hintRowUUIDs'].push([]) }
            pattern.header.push('hintRowUUIDs')
        }
    }

    // Find seed row UUIDs
    for(let i = 0; i < rowRatingsForCurrentPattern.length; i++) {
        if (
            rowRatingsForCurrentPattern[i].rating == RATING_CENTRAL || 
            rowRatingsForCurrentPattern[i].rating == RATING_CENTRALSW ||
            (rowRatingsForCurrentPattern[i].rating == RATING_GROUNDING && includeGrounding == true) ||
            (rowRatingsForCurrentPattern[i].rating == RATING_LEXGLUE && includeLexGlue == true)
            ) {

            uuidsOut.push(rowRatingsForCurrentPattern[i].row.uuid);

            // Also include hint row UUIDs, if enabled
            if (includeHintRows == true) {
                // NOTE: I forget whether rowRatingsForCurrentPattern and pattern are guaranteed to have the same order, so here we just 
                // manually search for the hint rows for each valid UUID.  This slows it down a bit, but the refresh should still generally
                // happen in less than about one second. 
                for (let j=0; j<pattern['UID'].length; j++) {
                    if (pattern['UID'][j] == rowRatingsForCurrentPattern[i].row.uuid) {
                        for (let k=0; k<pattern['hintRowUUIDs'][j].length; k++) {
                            uuidsOut.push(pattern['hintRowUUIDs'][j][k]);
                            // console.log("Including hint row: " + pattern['hintRowUUIDs'][j][k])
                        }
                    }
                }
            }
        }
    }

    return uuidsOut
}

// Returns the seed rows from the current selected pattern
function getSeedRows(){
    var rowsOut = []

    for(let i = 0; i < pattern.length; i++) {
        let uid = pattern['UID'][i]
        if(!isRowAHintInAnotherRow(uid)) {
            if (isSeedRow(pattern['RATING'][i])) {
                rowsOut.push(getTextDelByUID(uid));
            }
        }
    }

    return rowsOut
}

// Returns the seed rows from the current selected pattern
function getSeedRowsPkg(){
    var rowsOut = []

    for(let i = 0; i < pattern.length; i++) {
        if (pattern['RATING'][i] == RATING_CENTRAL || pattern['RATING'][i] == RATING_CENTRALSW || pattern['RATING'][i] == RATING_GROUNDING) {
            let uid = pattern['UID'][i]
            let text = getTextDelByUID(uid)

            // Check that this row is not part of another node
            if(!isRowAHintInAnotherRow(uid)) {

                // Create a list of the hint rows
                let hintRowUUIDs = pattern['hintRowUUIDs'][i]
                let hintRows = []
                for(let hintRowUUID of hintRowUUIDs){
                    hintRows.push({
                        text: tableMap[hintRowUUID],
                        uid: hintRowUUID,
                    })
                }
                
                rowsOut.push({
                    text: text,
                    uid: uid,
                    hintRows: hintRows,
                });
            }
            else{
                console.log(uid)
            }
        }
    }

    return rowsOut
}

// Returns the seed rows from the current selected pattern
function getSeedPkgAnnotated(){
    var rowsOut = []

    for(let i = 0; i < pattern.length; i++) {
        if (pattern['RATING'][i] == RATING_CENTRAL || pattern['RATING'][i] == RATING_CENTRALSW || pattern['RATING'][i] == RATING_GROUNDING) {
            let uid = pattern['UID'][i]

            // Check that this row is not part of another node
            if(!isRowAHintInAnotherRow(uid)) {

                // Create a list of the hint rows
                let hintRowUUIDs = pattern['hintRowUUIDs'][i]
                let hintRows = []
                for(let hintRowUUID of hintRowUUIDs){
                    hintRows.push({
                        tableRow: getTableRow(hintRowUUID),
                        uid: hintRowUUID,
                    })
                }
                
                rowsOut.push({
                    tableRow: getTableRow(uid),
                    uid: uid,
                    hintRows: hintRows,
                });
            }
        }
    }

    return rowsOut
}

// Rows in the pattern that go in the graph
function getApprovedRows(){
    var rowsOut = []

    for(let i = 0; i < pattern.length; i++) {
        let role = pattern['RATING'][i]
        if (role == RATING_CENTRAL || role == RATING_CENTRALSW || role == RATING_GROUNDING || role == RATING_LEXGLUE) {

            let uid = pattern['UID'][i]

            // Check that this row is not part of another node
            if(!isRowAHintInAnotherRow(uid)) {

                // Create a list of the hint rows
                let hintRowUUIDs = pattern['hintRowUUIDs'][i]
                let hintRows = []
                for(let hintRowUUID of hintRowUUIDs){
                    hintRows.push({
                        tableRow: getTableRow(hintRowUUID),
                        uid: hintRowUUID,
                    })
                }
                
                let tableRow = getTableRow(uid)
                rowsOut.push({
                    tableRow: tableRow,
                    uid: uid,
                    hintRows: hintRows,
                });
            }
        }
    }

    return rowsOut
}

// Returns the seed rows from the current selected pattern
function getSeedLemmas(){
    var lemmas = new Set()

    for(let i = 0; i < pattern.length; i++) {
        if (pattern['RATING'][i] == RATING_CENTRAL || pattern['RATING'][i] == RATING_CENTRALSW || pattern['RATING'][i] == RATING_GROUNDING) {
            let uid = pattern['UID'][i]

            let words = pattern['ROW'][i].split(" ")
            for(word of words){
                if(!stopWords.includes(word)) lemmas.add(lemmatize(word))
            }
        }
    }

    return lemmas
}

// Add a row to the master list of rows to rate
function addRowToRate(uuid, count, rating, isOptional) {
    var row = findRowByUUID(uuid)

    var rowRating = {
        row: row,
        rating: rating,
        count: count,
        optional: isOptional,
    }

    if(row.tableName === 'N/a'){
        rowRating.rating = RATING_BAD
    }
    
    rowRatingsForCurrentPattern.push(rowRating)
}

// Adds a row to the pattern datastructure
function appendRowToPattern(uid, rating, count, hintRowUUIDs, hintWords, rowNotes, isOptional) {
    
    // Get frequencies from the distrobution
    let freqs = rowRoleFreqs[uid]

    let background = 0
    let central = 0
    let grounding = 0
    let lexglue = 0
    let role = 0

    // This means there is no info on this row, dist needs to be updated
    if(typeof freqs === 'undefined'){
        console.log(`%cWarning! No role frequencies for row: "${uid}"`, 'color:orange')
    }
    else{
        background = freqs['BACKGROUND']
        central = freqs['CENTRAL']
        grounding = freqs['GROUNDING']
        lexglue = freqs['LEXGLUE']
        role = freqs['ROLE']
    }

    // get the table name and row text for this uid
    let row = getTableRow(uid)
    let textDel = getTextDelFromTableRow(row)

    if (!pattern.hasOwnProperty('lastCount')) {
        let unpopulatedCounts = Array(pattern.length).fill(-1)        
        pattern['lastCount'] = unpopulatedCounts        
        pattern.header.push('lastCount')
    }

    if (!pattern.hasOwnProperty('hintRowUUIDs')) {                
        pattern['hintRowUUIDs'] = [];
        for (let a=0; a<pattern.length; a++) { pattern['hintRowUUIDs'].push([]) }
        pattern.header.push('hintRowUUIDs')
    }
    
    if (!pattern.hasOwnProperty('hintWords')) {        
        pattern['hintWords'] = Array(pattern.length).fill([])
        pattern.header.push('hintWords')
    }

    if (!pattern.hasOwnProperty('rowNotes')) {        
        pattern['rowNotes'] = Array(pattern.length).fill("")
        pattern.header.push('rowNotes')
    }

    if (!pattern.hasOwnProperty('OPTIONAL')) {        
        pattern['OPTIONAL'] = Array(pattern.length).fill(false)
        pattern.header.push('OPTIONAL')
    }

    pattern["BACKGROUND_count"].push(background)
    pattern["CENTRAL_count"].push(central)
    pattern["GROUNDING_count"].push(grounding)
    pattern["LEXGLUE_count"].push(lexglue)
    pattern["ROLE_count"].push(role)
    pattern["RATING"].push(rating)
    pattern["ROW"].push(textDel)
    pattern["TABLE"].push(row.tablename)
    pattern["UID"].push(uid)
    pattern["length"] += 1
    pattern["lastCount"].push(count)
    pattern["hintRowUUIDs"].push(hintRowUUIDs)
    pattern["hintWords"].push(hintWords)
    pattern["rowNotes"].push(rowNotes)
    pattern["OPTIONAL"].push(isOptional)

    console.log(pattern)
}

function setCountInPattern(uuid, count) {
    var index = pattern["UID"].indexOf(uuid)
    if (index >= 0) {
        // Ensure that the 'lastCount' property exists
        if (!pattern.hasOwnProperty('lastCount')) {
            let unpopulatedCounts = Array(pattern.length).fill(-1)        
            pattern['lastCount'] = unpopulatedCounts        
            pattern.header.push('lastCount')
        }

        // Set the count
        pattern["lastCount"][index] = count;
        changesMadeSinceLastSave = true
    } else {
        console.log("ERROR: UUID not found in pattern (" + uuid + ").")
    }
}

// Send the pattern state to the server
// Adding a newName will delete the pattern of the current name and save it as this name
function save(newName = '', copy=false, whereFrom="") {
    // console.log("save() called from: " + whereFrom);

    let selectedDataset = getSelectedOptionValue("datasetSelector")
    if(selectedDataset == "ERR") return

    let patternName = patternNames[patternIdx]

    let edgeTable = getEdgeTableFromGraph()

    let autoGenIML = createIMLFromPattern()
    let patternIML = (dataInEditor)? editorIML.getValue() : 
                    (pattern.iml.length > 0)? pattern.iml : 
                    createIMLFromPattern()

    let data = {
        pattern: pattern,
        patternName: patternName,
        edgeTable: edgeTable,
        patternIML: patternIML,
        autoGenIML: autoGenIML,
        dataset: selectedDataset,
        newName: newName,
        copy: copy,
    }

    // console.log(data)

    // Stringify first
    let dataStr = JSON.stringify(data)
    let coolData = {
        dataString: dataStr
    }

    // console.log(`Sending approx. : ${parseInt(dataStr.length / 1000)} KBytes of data`)
    $.post("/PostPattern", coolData, function(res, status){
        console.log(res)
        changesMadeSinceLastSave = false
    })
}

function DEBUGdisplayPattern(str) {
    console.log("DEBUG " + str);
    console.log(pattern);
}

function reConstraintPattern(){

    let safeToReconstrain = true
    let autoGenIML = createIMLFromPattern()
    if(pattern.iml.length > 0 && pattern.iml != autoGenIML) {
        safeToReconstrain = confirm("Warning! Changes to IML are being overwritten.")
    }

    if(safeToReconstrain){

        pattern.iml = autoGenIML
        
        editorIML.setValue(autoGenIML)
    }
}

// Handles updating the idx and setting html
function selectPattern(selectedPattern, idx=-1){

    try {
        // Update the patternSelection stack
        patternSelectionStack.push(selectedPattern)
        
        // Focus the first row
        currentRow = "row_0"

        // Update the selected pattern idx & pattern
        if(idx >= 0){
            patternIdx = idx
        }
        else{
            patternIdx = getPatternIdx(selectedPattern)
        }
        
        let patternName = patternNames[patternIdx]
        
        // If this function is ever called outside of the selector callback
        // we need this set manually
        // if it was a callback call then this will be redundant
        selectByValue('patternSelector', patternName)

        // read the pattern reference from the list of patterns
        pattern = patterns[patternIdx]

        console.log("PATTERN IDX: " + patternIdx)
        print(pattern)

        // Check to see if 'lastCount' property exists -- if not, populate with dummy counts that will shortly be regenerated below.
        if (!pattern.hasOwnProperty('lastCount')) {
            let unpopulatedCounts = Array(pattern.length).fill(-1)
            pattern['lastCount'] = unpopulatedCounts
            pattern.header.push('lastCount')
        }

        if (!pattern.hasOwnProperty('hintRowUUIDs')) {        
            pattern['hintRowUUIDs'] = [];
            for (let a=0; a<pattern.length; a++) { pattern['hintRowUUIDs'].push([]) }
            pattern.header.push('hintRowUUIDs')
        }
        
        if (!pattern.hasOwnProperty('hintWords')) {        
            pattern['hintWords'] = Array(pattern.length).fill([])        
            pattern.header.push('hintWords')
        }

        if (!pattern.hasOwnProperty('rowNotes')) {        
            pattern['rowNotes'] = Array(pattern.length).fill("")        
            pattern.header.push('rowNotes')
        }

        if (!pattern.hasOwnProperty('OPTIONAL')) {        
            pattern['OPTIONAL'] = Array(pattern.length).fill(false)        
            pattern.header.push('OPTIONAL')
        }

        // Populate the graph field
        // If the tables is on file use that, else regenerate the graph
        if(edgeTablesMap.hasOwnProperty(patternName)){
            graphFromEdgeTable()
        }
        else{
            patternToGraph()
        }

        // Initialize the notes field
        document.getElementById('userNotes').value = pattern['notes']

        // Initialize the flag buttons
        styleFlagButtons()
        styleMobilityButtons()

        // Update the pattername header
        let uncleanName = patternNames[patternIdx]
        let cleanName = uncleanName.replace(/_/g, ' ')
        setPatternName(cleanName)
    
        // Determine the curent seed rows 
        var seedRows = [];
        for (var i=0; i<pattern.length; i++) {
            let uid = pattern['UID'][i]
            let rating = pattern['RATING'][i]
            if ((rating == RATING_CENTRAL) || (rating == RATING_CENTRALSW)) {
                seedRows.push(uid);
            }
        }

        // Find potential pattern rows by querying the question explanations
        let resultUUIDs = querySeedRows(seedRows)

        // Populate the rowRatings datastructure that the html uses
        rowRatingsForCurrentPattern = []
        for(let i = 0; i < pattern.length; i++){
            let uuid = pattern['UID'][i]
            let rating = pattern['RATING'][i]
            let isOptional = pattern['OPTIONAL'][i]
            //let count = getCountForRow(uid)         //## INCORRECT
            let count = pattern['lastCount'][i]         // ## POSSIBLY CORRECT
            if (count == -1) {                  
                // If the count hasn't been previously populated, then generate it now
                count = resultUUIDs.get(uuid)
            }
            // If the count is undefined, then set it to zero
            if (count === 'undefined') {
                count = 0;
            }
            //let count = 5;

            addRowToRate(uuid, count, rating, isOptional)
        }

        // update the HTML
        populatePatternHTML()

        // Update the questions window
        populateQuestionsWindow()

        // Update the overlap window
        calculateHintOverlap()

    }
    catch (err) {
        alert("Failed to load pattern" + err)
        console.log(err)

    }
}

function toggleMarkerFilter(str){

    switch(str){
        case 'DONE':
            showDone = (showDone)? false : true;
            break;

        case 'GOOD':
            showGood = (showGood)? false : true;
            break;
        
        case 'UNCERTAIN':
            showUncertain = (showUncertain)? false : true;
            break;

        case 'BAD':
            showBad = (showBad)? false : true;
            break;
        
        case 'REDUNDANT':
            showRedundant = (showRedundant)? false : true;
            break;
    
        case 'UNMARKED':
            showUnmarked = (showUnmarked)? false : true;
            break;
            
        default:
            showDone = true
            showGood = true
            showUncertain = true
            showBad = true
            showRedundant = false
            showUnmarked = true
    }

    // Style the buttons
    styleFilterButtons()

    // Update the selector
    populatePatternSeletor()

    // Update the pattern selected
    patternChanged()
}

/* 
 * Polling Functions
 */

// Polling function
function beginPolling() {
    // console.log('beginPolling(')
    setTimeout(poll, POLLING_INTERVAL)
}

// Run these every poll
function poll(){
    // Cooldown that prevent polling before a certain point
    if(READY_TO_POLL){

        // ping the server
        if(PING_TIMER > 0){
            PING_TIMER --
    
            if(PING_TIMER === 0){
                $.get('/Ping')
                .done(function(res, status){
                    document.getElementById('upTimeButton').classList.remove('disabled');
                    document.getElementById('upTimeButton').classList.add('unclicked');
                    document.body.style.backgroundColor = "#FFFFFF";
                    PING_TIMER = PING_CYLES
                })
                .fail(function(xhr, status, error){                
                    document.getElementById('upTimeButton').classList.remove('unclicked');
                    document.getElementById('upTimeButton').classList.add('disabled');
                    document.body.style.backgroundColor = "#ff8c94";
                    PING_TIMER = PING_CYLES
                })
            }
        }

        // Publishing button update
        // Just decrement over time and call the styling button when its off cooldown
        if(PUBLISH_COOLDOWN > 0){
            PUBLISH_COOLDOWN--
    
            if(PUBLISH_COOLDOWN == 0) stylePublishingButton()
        }
        
        // if pattern is not set dont run any of this
        if(pattern !== null){
    
            // If the value is different then something has changed
            let notesHaveChanged = false
            if(document.getElementById('userNotes').value !== pattern['notes']) {
                notesHaveChanged = true
                // console.log(pattern['notes'])
            }
        
            // Update our notes var
            pattern['notes'] = document.getElementById('userNotes').value
        
            // Save if changes made
            // also update stats here
            if((changesMadeSinceLastSave || notesHaveChanged)) {
                
                // Save if autosave enabled
                if(AUTOSAVE == true) {
                    console.log("Autosaving...")
                    save(name="", copy=false, whereFrom="AutoSave")
                }
    
                // 
                updateStatistics()
        
                // Update the overlap window
                calculateHintOverlap()
            }
        }
    }

    setTimeout(poll, POLLING_INTERVAL)
}

/*
 * Constraint Utils
 */

function getRoleCode(role){

    let code = ''

    switch(role){
        case 'CENTRAL':
            code = 'C'
            break
        case 'CENTRALSW':
            code = 'SW'
            break
        case 'GROUNDING':
            code = 'G'
            break
        case 'MAYBE':
            code = 'M'
            break
        case 'LEXGLUE':
            code = 'LG'
            break
        case 'BAD':
            code = 'B'
            break
        default:
            code = 'X'
            break
    }

    return code
}

// Check that these two rows meet the given constraints
function rowsMeetConstraints(uid_i, uid_j, constraints){
    // console.log(`rowsMeetConstraints(${text_i}, ${text_j}, constraints)`)

    let text_i = tableMap[uid_i]
    let text_j = tableMap[uid_j]

    let cols_i = text_i.split(' | ')
    let cols_j = text_j.split(' | ')

    let numConstraintsMet = 0
    for(let {colIdx_from, colIdx_to, lemmas} of constraints){
        // let lemmaSet = new Set(lemmas)
        let col_i = cols_i[colIdx_from]
        let col_j = cols_j[colIdx_to]
        
        // TODO this is happening for some reason on pattern 3
        // FIXME
        if(typeof col_i == 'undefined'){
            // console.log('BADF00D_i')
            // console.log(cols_i)
            // console.log(cols_j)
            // console.log(constraint)
            return false
        }
        if(typeof col_j == 'undefined'){
            // console.log('BADF00D_j')
            // console.log(cols_i)
            // console.log(cols_j)
            // console.log(constraint)
            return false
        }
        
        // let words_i = new Set(tokenize(col_i))
        // let words_j = new Set(tokenize(col_j))
        // let ovlCount_i = intersection(words_i, lemmaSet).length
        // let ovlCount_j = intersection(words_j, lemmaSet).length
        // let constraintMet = ovlCount_i == ovlCount_j && ovlCount_j == lemmas.length

        let words_i = tokenize(col_i)
        let words_j = tokenize(col_j)

        let constraintMet = false
        for(word_i of words_i){
            let lemma_i = lemmatize(word_i)
            for(word_j of words_j){
                let lemma_j = lemmatize(word_j)

                if(lemma_i == lemma_j){
                    constraintMet = true
                }
            }   
        }

        if(constraintMet) numConstraintsMet++
    }

    if(numConstraintsMet == constraints.length) return true
    else if (numConstraintsMet == 0) return false
    else{
        // console.log(`Overconstrained
        //                 ${text_i} 
        //                 ${text_j}`)
        return true
    }
}

// return a flag list of which constraints were met
function rowsMeetWhichConstraints(uid_i, uid_j, constraints){

    let constraintsMet = []

    let text_i = tableMap[uid_i]
    let text_j = tableMap[uid_j]

    let cols_i = text_i.split(' | ')
    let cols_j = text_j.split(' | ')

    for(let n in constraints){
        let {colIdx_from, colIdx_to, lemmas} = constraints[n]

        // let lemmaSet = new Set(lemmas)
        let col_i = cols_i[colIdx_from]
        let col_j = cols_j[colIdx_to]
        
        // rows not in same table
        if(typeof col_i == 'undefined') return false
        if(typeof col_j == 'undefined') return false
        
        let words_i = tokenize(col_i)
        let words_j = tokenize(col_j)

        let constraintMet = false
        for(word_i of words_i){
            let lemma_i = lemmatize(word_i)
            for(word_j of words_j){
                let lemma_j = lemmatize(word_j)

                if(lemma_i == lemma_j){
                    constraintMet = true
                }
            }   
        }

        if(constraintMet) constraintsMet.push(constraints[n])
    }

    return constraintsMet
}

// ASSUME ANNOTATOR WRONG
// Mark the hint rows that meet the constraints
function markProblemHintRows(hintRows_i, hintRows_j, possibleEdges, edge){

    let node_src = edge.source()
    let node_tgt = edge.target()

    // Initialize
    for (let i in hintRows_i) {hintRows_i[i]['fits'] = false}
    for (let j in hintRows_j) {hintRows_j[j]['fits'] = false}

    // Raise flags for rows that fit the constraints
    for (let i in hintRows_i){
        let pkg_i = hintRows_i[i]
        let uid_i = pkg_i.uid
        
        for (let j in hintRows_j){
            let pkg_j = hintRows_j[j]
            let uid_j = pkg_j.uid

            // If constraints are met, then make the two hint rows
            if(rowsMeetConstraints(uid_i, uid_j, possibleEdges)){
                hintRows_i[i]['fits'] = true
                hintRows_j[j]['fits'] = true
            }
        }   
    }

    // emphasize the nodes and subsequent rows that arent meeting constraints  
    let isProblemOnEdge = false        
    for(let i in hintRows_i){
        let {text, uid, fits} = hintRows_i[i]
        if(!fits) {
            isProblemOnEdge = true 
            
            // Notify console
            // console.log(`** "${cleanTextDel(text)}" nofit`)

            // Classify the node as having a problem
            node_src.addClass('problem')

            // Clean delimited text
            let textClean = cleanTextDel(text)

            // Emphasise the problem row
            let lines = node_src.data('label').split('\n')
            for(let n in lines){
                let line = lines[n]
                if(line.includes(textClean) && !line.startsWith("***")){
                    lines[n] = `***${line}***`
                }
            }
            node_src.data('label', lines.join('\n'))
        }
    }     
    for(let i in hintRows_j){
        let {text, uid, fits} = hintRows_j[i]
        if(!fits) {

            isProblemOnEdge = true 

            // Notify console
            // console.log(`***"${text}" nofit`)

            // Classify the node as having a problem
            node_tgt.addClass('problem')

            // Clean delimited text
            let textClean = cleanTextDel(text)

            // Emphasise the problem row
            let lines = node_tgt.data('label').split('\n')
            for(let n in lines){
                let line = lines[n]
                if(line.includes(textClean) && !line.startsWith("***")){
                    lines[n] = `***${line}***`
                }
            }
            node_tgt.data('label', lines.join('\n'))
        }
    }

    if(isProblemOnEdge) edge.addClass('problem')
    else edge.removeClass('problem')
}

// ASSUME ANNOTATOR RIGHT
// Remove constraints that dont work for all hint rows
function removeProblemConstraints(hintRows_i, hintRows_j, possibleEdges, edge){

    // Initialize
    for (let i in hintRows_i) {hintRows_i[i]['fits'] = false}
    for (let j in hintRows_j) {hintRows_j[j]['fits'] = false}

    // Raise flags for rows that fit the constraints
    let constraintsSetInitYet = false
    let constraintsSet = new Set()
    for (let i in hintRows_i){
        let pkg_i = hintRows_i[i]
        let uid_i = pkg_i.uid
        
        let constraints_from_i = []
        for (let j in hintRows_j){
            let pkg_j = hintRows_j[j]
            let uid_j = pkg_j.uid

            // Find the uid in node j who meets the most constraints
            let constraints = rowsMeetWhichConstraints(uid_i, uid_j, possibleEdges)
            if(constraints.length > constraints_from_i.length) constraints_from_i = constraints
        }

        // Initialize once with the first set of constraints
        if(!constraintsSetInitYet) {
            constraintsSet = new Set(constraints_from_i)
            constraintsSetInitYet = true
        }
        // Then perform intersections each round
        else constraintsSet = intersection(constraintsSet, new Set(constraints_from_i))
    }

    // Go both ways
    for (let j in hintRows_j){
        let pkg_j = hintRows_j[j]
        let uid_j = pkg_j.uid
        
        let constraints_from_j = []
        for (let i in hintRows_i){
            let pkg_i = hintRows_i[i]
            let uid_i = pkg_i.uid

            // Find the uid in node j who meets the most constraints
            let constraints = rowsMeetWhichConstraints(uid_i, uid_j, possibleEdges)
            if(constraints.length > constraints_from_j.length) constraints_from_j = constraints
        }

        // Perform intersections each round
        constraintsSet = intersection(constraintsSet, new Set(constraints_from_j))
    }

    // The constraints left are the ones that all hint rows meet
    let constraints = [...constraintsSet]
    let text_from = tableMap[hintRows_i[0].uid]
    let text_to = tableMap[hintRows_j[0].uid]
    let label = createEdgeLabel(constraints, text_from, text_to)
    edge.data('constraints', constraints)
    edge.data('label', label)
    if(label.length == 0) edge.style('visibility', 'hidden')
}

// Determine the number of hint rows for this row in the current pattern
function getHintRows(uid){
    for(let i = 0; i < pattern.length; i++){
        if(pattern['UID'][i] == uid) return pattern['hintRowUUIDs'][i]
    }
    return []
}

// 
function getRoleShortForUIDInPattern(uid){
    for(let i = 0; i < pattern.length; i++){
        let uid_i = pattern['UID'][i]
        if(uid_i == uid){

            let role = pattern['RATING'][i]

            return getRoleCode(role)
        }
    }

    return 'unk'
}

// 
function getRoleForUIDInPattern(uid){
    for(let i = 0; i < pattern.length; i++){
        let uid_i = pattern['UID'][i]
        if(uid_i == uid){

            let role = pattern['RATING'][i]

            return role
        }
    }

    return 'unk'
}

// 
function getTableNameFromText(text){
    return text.substring(text.lastIndexOf('(')+1, text.lastIndexOf(','))
}

// requires uid to be a part of the current active pattern
function getTableNameFromPatternWithUID(uid){
    for(let i = 0; i < pattern.length; i++){
        let uid_i = pattern['UID'][i]
        let tableName = pattern['TABLE'][i]

        if(uid_i == uid){
            return tableName
        }
    }
    return ''
}

// searches tablestore for tablename
function getTableNameFromUID(uid){
    
    for(let {tableName, uuid} of tableRows){
        if(uid == uuid) return tableName
    }
    return "ERROR"
}

// "... | ... | (KINDOF, UID: 49dc-4526-46f9-5b32)" -> "49dc-4526-46f9-5b32"
function getUIDFromDelText(textDel){
    col = textDel.split(' | ').pop()
    uid = col.substring(col.indexOf('UID:')+6, col.length-1)
    return uid
}

// Tokenize a sentance 
function tokenize(sentance, withLemmatize=false){
    words = []

    // Remove non alpha-numerics
    words_almost = sentance.replace(/[^\w\s\-\']/g, ' ').split(' ')

    // Remove '' elements
    for( word_almost of words_almost ){
        if(word_almost.length > 0) {
            if(withLemmatize) words.push(lemmatize(word_almost))
            else words.push(word_almost)
        }
    }

    return words
}

// Boolean function for when the header is a data head
function isDataCol(head){
    return !head.startsWith('[FILL]') && !head.startsWith('[SKIP]')
}

// Set intersection on edge objects
function edgeIntersection(edgeSetA, edgeSetB){
    
    let newSet = edgeSetA.filter(function({col_from, col_to}){
        let setBHasThisConstraint = false
        for(let edge of edgeSetB){
            if(edge.col_from == col_from && edge.col_to == col_to) setBHasThisConstraint = true
        }
        return setBHasThisConstraint
    })
    return newSet
}

// Reads the class from the pattern hash of user edits
function getClassForEdge(classHash){
    let userIMLEdits = pattern.userIMLEdits
    if(userIMLEdits.hasOwnProperty(classHash))  return userIMLEdits[classHash].classes
    else                                        return CLASS_UNRATED
}

function mkConstraintEdge(colIdx_from, colIdx_to, on, lemmas, classes, classHash){
    return {
        colIdx_from: colIdx_from,
        colIdx_to: colIdx_to,
        on: on,
        lemmas: lemmas,
        classHash: classHash,
        classes: classes,
    }
}

function mkConstraintEdgeFromEdgeTableEntry(constraintStr){
    try{
        let cols = constraintStr.split(":")

        let colIdx_from = cols[0]
        let colIdx_to = cols[1]

        let classHash = 0//cols[4]
        let classes = getClassForEdge(classHash)
    
        return {
            colIdx_from: colIdx_from,
            colIdx_to: colIdx_to,
            on: cols[2] == 'Y',
            lemmas: cols[3].split(";"),
            classHash: 0,
            classes: classes,
        }
    }
    catch(err){
        console.log(constraintStr)
        console.log(cols)
    }
}

/*
// returns what edges can exist between two rows, taking into account hint rows
// Heuristic
function determinePossibleEdges(row_0, row_1){

    let uid_0 = row_0.uuid
    let uid_1 = row_1.uuid

    // Get hint rows
    let hintRowsMap = createHintRowsMap()
    let hintRows_0 = hintRowsMap[uid_0]
    let hintRows_1 = hintRowsMap[uid_1]

    // First determine edges from the root row to row
    let possibleEdges = determinePossibleEdges_helper(row_0, row_1)

    // Only continue if root rows have connection and we are removing overconstraints
    let toggleOverConstraint = document.getElementById("toggleOverConstraint").checked
    if(possibleEdges.length > 0 && !toggleOverConstraint){

        // Determine intersection of tightest constraints for each row in node_0 connecting to a row in node_1
        for (let pkg_i of hintRows_0){
            let row_i = pkg_i.tableRow
            let uid_i = pkg_i.uid
            
            // Determine longest possible edges from this row to the other possible hint rows
            // Assume the longest is the correct connection
            let longest_possibleEdges = []
            for (let pkg_j of hintRows_1){
                let row_j = pkg_j.tableRow
                let uid_j = pkg_j.uid
                
                // Determine possible constraints
                let possibleEdges_ij = determinePossibleEdges_helper(row_i, row_j)

                if(possibleEdges_ij.length > longest_possibleEdges.length) longest_possibleEdges = possibleEdges_ij
            }

            // Intersect to minimize constraints
            possibleEdges = edgeIntersection(possibleEdges, longest_possibleEdges)
        }
    }

    // Remove duplicate lemmas from each edge
    for(let i in possibleEdges){

        let lemmaSeen = new Set()       // Hash for determining duplicates
        let newLemmaList = []           // new list
        let {lemmas} = possibleEdges[i] // current lemmas
        for(let lemma of lemmas){
            if(!lemmaSeen.has(lemma)){
                lemmaSeen.add(lemma)
                newLemmaList.push(lemma)
            }
        }

        // Write to memory
        possibleEdges[i].lemmas = newLemmaList
    }

    return possibleEdges
}
*/

// returns what edges can exist between two rows, taking into account hint rows
// Heuristic
function determinePossibleEdges(row_0, row_1){

    let uid_0 = row_0.uuid
    let uid_1 = row_1.uuid

    // Get hint rows
    let hintRowsMap = createHintRowsMap()
    let hintRows_0 = hintRowsMap[uid_0]
    let hintRows_1 = hintRowsMap[uid_1]
    
    // First determine edges from the root row to row
    let possibleEdges = determinePossibleEdges_helper(row_0, row_1)
    let possibleEdges_over = deepCopy(possibleEdges)

    // Only continue if root rows have connection
    if(possibleEdges.length > 0){

        // Determine intersection of tightest constraints for each row in node_0 connecting to a row in node_1
        for (let pkg_i of hintRows_0){
            let row_i = pkg_i.tableRow
            
            // Determine longest possible edges from this row to the other possible hint rows
            // Assume the longest is the correct connection
            let longest_possibleEdges = []
            for (let pkg_j of hintRows_1){
                let row_j = pkg_j.tableRow
                
                // Determine possible constraints
                let possibleEdges_ij = determinePossibleEdges_helper(row_i, row_j)

                if(possibleEdges_ij.length > longest_possibleEdges.length) longest_possibleEdges = possibleEdges_ij
            }

            // Intersect to minimize constraints
            possibleEdges = edgeIntersection(possibleEdges, longest_possibleEdges)
        }
    }

    // Remove duplicate lemmas from each edge
    for(let i in possibleEdges){

        let lemmaSeen = new Set()       // Hash for determining duplicates
        let newLemmaList = []           // new list
        let {lemmas} = possibleEdges[i] // current lemmas
        for(let lemma of lemmas){
            if(!lemmaSeen.has(lemma)){
                lemmaSeen.add(lemma)
                newLemmaList.push(lemma)
            }
        }

        // Write to memory
        possibleEdges[i].lemmas = newLemmaList
    }

    return [possibleEdges, possibleEdges_over]
}

// returns what edges can exist between two rows
function determinePossibleEdges_helper(row_0, row_1){
   
    let possibleEdges = []

    let cells_0 = row_0.cellLemmas
    let cells_1 = row_1.cellLemmas

    let cellTags_0 = row_0.cellTags

    // read the headers so we know where the skip/fill rows are
    let row_header_0 = row_0.header
    let row_header_1 = row_1.header

    // Comparing two 3d spaces for lemma overlap

    // Cell level (Check that each cell is a data col)
    for(let i = 0; i < cells_0.length; i++){
        let head_0 = row_header_0[i]
        if(isDataCol(head_0)){
            let cell_0 = cells_0[i]
            for(let j = 0; j < cells_1.length; j++){
                let head_1 = row_header_1[j]
                if(isDataCol(head_1)){
                    let cell_1 = cells_1[j]

                    // Create a list of all overlapping lemmas
                    let ovlLemmas = []
                    
                    // Alternative level
                    for(let n = 0; n < cell_0.length; n++){
                        let lemmas_0 = cell_0[n]
                        for(let m = 0; m < cell_1.length; m++){
                            let lemmas_1 = cell_1[m]
                        
                            // Lemma level (Skip non content tags)
                            for(let ii = 0; ii < lemmas_0.length; ii++){
                                let lemma_0 = lemmas_0[ii]
                                if(isContentTag(cellTags_0[i][n][ii])){
                                    for(let jj = 0; jj < lemmas_1.length; jj++){
                                        let lemma_1 = lemmas_1[jj]
                                        
                                        // If lemma match save the lemma
                                        if(lemma_0 == lemma_1) {
                                            ovlLemmas.push(lemma_0)
                                        }
                                    }   
                                }
                            }
                        }   
                    }

                    // if there is more than one lemma then save the constraint on the stack
                    if(ovlLemmas.length > 0){
                        possibleEdges.push(mkConstraintEdgeFromEdgeTableEntry(`${i}:${j}:Y:${ovlLemmas.join(";")}:0`))
                    }
                }
            }   
        }
    }

    return possibleEdges 
}

// Get a map of all the hint rows keyed by uid
// {
//    xxxx-xxxx-xxxx-xxxx: [
//                             {
//                                 uid: xxxx-xxxx-xxxx-xxxx,
//                                 text: "...",
//                                 tableRow: <TableRow>,
//                             },
//                             {
//                                 uid: yyyy-yyyy-yyyy-yyyy,
//                                 text: "...",
//                                 tableRow: <TableRow>,
//                             },
//    ],
//    ...
// }
function createHintRowsMap(includeRootRow=true){
    let hintRowsMap = {}

    for(let i = 0; i < pattern.length; i++){
        let uid = pattern['UID'][i]
        let text = pattern['ROW'][i]
        let hintRowUUIDs = pattern['hintRowUUIDs'][i]

        // Initialize list of hint rows 
        let hintRows = []
        
        if(includeRootRow){
            hintRows = [
                {
                    uid: uid,
                    text: text,
                    tableRow: getTableRow(uid),
                }
            ]
        }
        
        for(let hintRowUUID of hintRowUUIDs){
            
            let tableRow = getTableRow(hintRowUUID)
            hintRows.push({
                uid: hintRowUUID,
                text: getTextDelFromTableRow(tableRow),
                tableRow:tableRow,
            })
        }

        hintRowsMap[uid] = hintRows
    }

    return hintRowsMap
}

function createRatingsMap(){
    let ratingsMap = {}

    for(let i = 0; i < pattern.length; i++){
        let uid = pattern['UID'][i]
        let rating = pattern['RATING'][i]

        ratingsMap[uid] = rating
    }

    return ratingsMap
}

/*
 * Graph Cytoscape
 */

function constraintsFromEdgeTableStr(fullConstraintStr){
    let constraints = []
    
    if(fullConstraintStr.length > 0){
        let constraintStrs = fullConstraintStr.split(',')
        for(constraintStr of constraintStrs){
            constraints.push(mkConstraintEdgeFromEdgeTableEntry(constraintStr))
        }
    }
    return constraints
}

function getEdgeTableStrFromConstraints(constraints){
    let fullEdgeTableStr = ""
    for(let {colIdx_from, colIdx_to, on, lemmas, classHash} of constraints){
        let lemmasStr = lemmas.join(';')
        fullEdgeTableStr += (on)? `${colIdx_from}:${colIdx_to}:Y:${lemmas.join(";")}:${classHash},` : `${colIdx_from}:${colIdx_to}:N:${lemmas.join(";")}:${classHash},`
    }
    // Trim comma
    let len = fullEdgeTableStr.length
    if(len > 0) fullEdgeTableStr = fullEdgeTableStr.substring(0, len-1)

    return fullEdgeTableStr
}

function getEdgeTableFromGraph(){
    let edgeTable = {}

    edgeTable['isDone'] = false


    let nodes = cy.nodes()
    let numNodes = nodes.length

    edgeTable['length'] = numNodes
    edgeTable['width'] = numNodes

    // Initialize the header as well as the frame structure
    let header = []
    for(let i = 0; i < numNodes; i++){
        // Populate header
        let uid_i = nodes[i].id()
        header.push(uid_i)

        edgeTable[uid_i] = {}
        for(let j = 0; j < numNodes; j++){
            let uid_j = nodes[j].id()
            edgeTable[uid_i][uid_j] = ""
        }
    }
    edgeTable['header'] = header
    edgeTable['index'] = header

    // Loop through the edges and store their constraints in the table  
    let edges = cy.edges()
    for(let i = 0; i < edges.length; i++){
        let edge = edges[i]
        let uid_src = edge.data('source')
        let uid_tgt = edge.data('target')
        let constraints = edge.data('constraints')

        edgeTable[uid_src][uid_tgt] = getEdgeTableStrFromConstraints(constraints)
    }

    return edgeTable
}

// Mark any issues present with the given node
function markIssuesOnNode(node){
    let uid = node.id()

    // Get all lemmas in pattern (root row only) (skip this node)
    let seedRowPkgs = getApprovedRows()
    let fullLemmaSet = new Set()
    for(let {tableRow} of seedRowPkgs){
        if(tableRow.uuid != uid){
            let lemmas = getLemmaSetFromRow(tableRow)
            fullLemmaSet = union(fullLemmaSet, lemmas)
        }
    }

    // For each hint row on this node: determine the lemma overlap with the full set
    let hintRowsMap = createHintRowsMap(includeRootRow=true)
    let hintRows = hintRowsMap[uid]

    let root_row = hintRows[0].tableRow
    
    let maxOvlIdx = -1
    let maxLemmasOvl = -1
    for(let i in hintRows){
        let {tableRow} = hintRows[i]

        let lemmas = getLemmaSetFromRow(tableRow)
        
        let numLemmasOvl = intersection(lemmas, fullLemmaSet).size

        if(numLemmasOvl > maxLemmasOvl){
            maxLemmasOvl = numLemmasOvl
            maxOvlIdx = i
        }
    }

    // If a hint row has the most overlap then flag that hint row
    if(maxOvlIdx != 0){

        // Classify the node as having a problem
        node.addClass('warning')
        node.data("warningRowIdxs", [maxOvlIdx-1])

        // Emphasise the problem row
        let lines = node.data('label').split('\n')
        for(let n in lines){
            if(n == parseInt(maxOvlIdx) + 1){
                lines[n] = `(WARNING) ${lines[n]}`
            }
        }
        node.data('label', lines.join('\n'))
    }
}

function updateNodeState(uid, prevRating='', rating='', optional=false, newUID=""){
    let toggleOverConstraint = document.getElementById("toggleOverConstraint").checked;
    // console.log(`updateNodeState(uid="${uid}", prevRating='${prevRating}', rating='${rating}', optional=${optional}, newUID="${newUID}")`)
    let row = getTableRow(uid)

    // Clear the problem class markers at the start
    cy.nodes().removeClass('problem')
    cy.edges().removeClass('problem')

    // Rating change
    if(prevRating.length > 0){
        // Update the graph based on the vote change
        let prevASeedRow = isApprovedRow(prevRating)
        let currASeedRow = isApprovedRow(rating)
    
        // If a new node was created add it
        if(!prevASeedRow && currASeedRow){
            
            // Add node
            addNodeToGraph(uid)
    
            // Determine possible edges and add them
            let hintRowsMap = createHintRowsMap()
            let hintRows_i = hintRowsMap[uid]
            let seedRows = getApprovedRows()
            for(let j = 0; j < seedRows.length; j++){
                let pkg_j = seedRows[j]
                let row_j = pkg_j.tableRow
                let uid_j = pkg_j.uid
                let hintRows_j = pkg_j.hintRows
                hintRows_j.unshift({tableRow:row_j,uid:uid_j})    // Add the base row to the hint rows list
                
                if(uid_j != uid){
                    // Get possible edges between these rows, if they exist then draw them
                    // * NOTE: we flip the uid direction so it ligns up with the edge table's convention
                    let [possibleEdges, possibleEdges_over] = determinePossibleEdges(row_j, row)
                    if(possibleEdges.length > 0){
                        
                        let edge = addEdgeToGraph(uid_j, uid, possibleEdges, constraints_over=possibleEdges_over)
        
                        // If we are overconstraining, dont remove the problems
                        if(toggleOverConstraint) markProblemHintRows(hintRows_j, hintRows_i, possibleEdges_over, edge)

                        // Determine if there are any hint rows that do not meet the constraints
                        // *** This should be redundant now
                        else removeProblemConstraints(hintRows_j, hintRows_i, possibleEdges, edge)
                    }
                }
            }   
        }
        // If a node was demarked, remove it
        else if(prevASeedRow && !currASeedRow){
            let node = cy.nodes(`#${uid}`)[0]
            cy.remove(node)
        }
        // If a node had its class changed update that class
        else if(prevASeedRow && currASeedRow){
            let node = cy.nodes(`#${uid}`)[0]
            
            node.removeClass(getRoleCode(prevRating))
            node.addClass(getRoleCode(rating))
        } 
        // node was not in graph and still should not be
        else{
            // No-Op
        }
    }
    // Check for new UID
    else if(newUID.length > 0){

        // Remove the current node
        // (Cytoscape returns a collection, there should only be one element)
        let node = cy.nodes(node => node.id() == uid)
        if(node.length == 1){
            node = node[0]
    
            // Add the new node with the updated UID
            let nodeNew = addNodeToGraph(newUID)

            // Move the Edges to the new node
            let edges_src = cy.edges(edge => edge.data('source') == uid)
            let edges_tgt = cy.edges(edge => edge.data('target') == uid)
            for(let i = 0; i < edges_src.length; i++){
                let edge = edges_src[i]
                edge.move({'source': newUID})
            }
            for(let i = 0; i < edges_tgt.length; i++){
                let edge = edges_tgt[i]
                edge.move({'target': newUID})
            }

            // Remove the old node
            cy.remove(node)

            // Mark any potential issues on the new node
            markIssuesOnNode(nodeNew)

            changesMadeSinceLastSave = true
        }
        else{
            console.log(`%c\tERROR! There are ${node.length} nodes for uid: "${uid}" in the graph.`, 'color:red')
            throw new Error(`ERROR! There are ${node.length} nodes for uid: "${uid}" in the graph.`)
        }

    }
    // Row was marked optional/not-optional
    else{
        // No-op
    }

    runLayout()
}

// Creates the label for a node 
function createNodeLabel(uid, textDel, tableName){
    
    let text = cleanTextDel(textDel)
    let label = `${tableName}\n${text}`

    // Include potential hint rows in the node
    let uids_hint = getHintRows(uid)  
    for(uid_hint of uids_hint){
        let text = cleanTextDel(tableMap[uid_hint])
        label += `\n${text}`
    }

    return label
}

// Creates the label for an edge 
function createEdgeLabel(constraints, text_from, text_to){
    let label = ''
    for(let {colIdx_from, colIdx_to, on} of constraints){
        if(on) label += `${text_from.split(' | ')[colIdx_from]} [${colIdx_from}] <-> ${text_to.split(' | ')[colIdx_to]} [${colIdx_to}]\n`
    }

    return label
}

// callback for when a node's hint rows are changed
// - Update label
// - Re-Highlight broken constraints
function updateHintRowsOnNode(uid){

    // Check that node is present on graph
    let node = cy.nodes(`#${uid}`)[0]
    if(typeof node != 'undefined'){
    
        // Clear the problem class markers at the start
        cy.nodes().removeClass('problem')
        cy.edges().removeClass('problem')
    
        // Generate new node label
        let textDel = tableMap[uid]
        let tableName = getTableNameFromPatternWithUID(uid)
        let newLabel = createNodeLabel(uid, textDel, tableName)

        // Update node label
        node.data('label', newLabel)
    
        // Re-check the edges for conflicts with current hint rows
        let edges = cy.edges(function(edge){
            return edge.data('source') == uid || edge.data('target') == uid
        })
    
        let hintRowsMap = createHintRowsMap()
    
        for(let i = 0; i < edges.length; i++){
            let edge = edges[i]
    
            let uid_src = edge.data('source')
            let uid_tgt = edge.data('target')
            let constraints = edge.data('constraints')
    
            markProblemHintRows(hintRowsMap[uid_src], hintRowsMap[uid_tgt], constraints, edge)
            // removeProblemConstraints(hintRowsMap[uid_src], hintRowsMap[uid_tgt], constraints, edge)
        }
    
    }
    else{
        console.log(`%cWarning! No node, on graph, with uid: "${uid}" to update.\nNo action performed.`, "color:orange")
    }
}

function addNodeToGraph(uid){

    let role = getRoleShortForUIDInPattern(uid)
    let role_full = getRoleForUIDInPattern(uid)
    
    let textDel = tableMap[uid]

    let cols = textDel.split(" | ")
    let uidCol = cols.pop()  // remove last col
    let tableName = getTableNameFromText(uidCol)

    let label = createNodeLabel(uid, textDel, tableName)

    // add node to graph through the API
    let node = cy.add({
        group: 'nodes',
        data: {
            id: uid,
            label: label,
            role: role_full,
        },
        classes: role,
    })[0]

    return node
}

function addEdgeToGraph(uid_from, uid_to, constraints, constraints_over=[]){

    // Determine edge class
    let srcRole = getRoleShortForUIDInPattern(uid_from)
    let tgtRole = getRoleShortForUIDInPattern(uid_to)
    let isHardEdge = srcRole == 'G' || srcRole == 'SW' || tgtRole == 'G' || tgtRole == 'SW'
    let classes = (isHardEdge)? '' : ' soft'

    // Generate edge label
    let text_from = tableMap[uid_from]
    let text_to = tableMap[uid_to]
    
    let toggleOverConstraint = document.getElementById('toggleOverConstraint').checked
    let label = createEdgeLabel(constraints, text_from, text_to)
    // console.log(`label "${label}"`)

    let edgeID = `${uid_from}.${uid_to}`

    // Add an edge ID to each constraint so they know where their mom is
    for(let i = 0; i < constraints.length; i++) constraints[i]['edgeID'] = edgeID

    // Add the edge
    let edge = cy.add({
        group: 'edges',
        data: {
            id: edgeID,
            label: label,
            source: uid_from,                       // Source node id
            target: uid_to,                       // Target node id
            constraints: constraints,
            constraints_over: constraints_over,
        },
        classes: `autorotate ${classes}`,                      // Controls how the label is displayed
    })[0]

    // Edge has no label, make it invisble
    if(label.length == 0){
        edge.style('visibility', 'hidden')
    }

    return edge
}

function runLayout(){

    // Populate edgeTargetLength
    cy.edges().forEach(function (edge) {
        var edgeLength = 1;        
        var nodeSource = edge.source()
        var nodeTarget = edge.target()

        if ((nodeSource.degree() == 1) || (nodeTarget.degree() == 1)) {
            edgeLength = 200;
        } else {
            edgeLength = 100 + ((nodeSource.degree() + nodeTarget.degree()) * 20);
        }

        edge.data('myTargetLength', edgeLength)
    });

    // Layout
    cy.layout({
        name: 'cose-bilkent',
        animate: 'end',
        //animationEasing: 'ease-out',
        animationDuration: 1200,
        randomize: false,
        idealEdgeLengthFunction: function(edge) {                    
            if (edge.data() && edge.data().myTargetLength) {
                return edge.data().myTargetLength;
            } else {
                return 300;
            }            
        },
        // Whether to include labels in node dimensions. Useful for avoiding label overlap
        nodeDimensionsIncludeLabels: true,

        // Node repulsion (non overlapping) multiplier
        nodeRepulsion: 1000000,
        // Divisor to compute edge forces
        edgeElasticity: 0.95,
        // Nesting factor (multiplier) to compute ideal edge length for inter-graph edges
        nestingFactor: 0.01,
        // Gravity force (constant)
        gravity: 10.95,
        // Maximum number of iterations to perform
        numIter: 10000,
        // number of ticks per frame; higher is faster but more jerky
        refresh: 1000,
        // Whether to fit the network view after when done
        fit: true,
    }).run()
}

// Generate inital automated edge table for graph
function patternToGraph(){
    console.log("patternToGraph()")
    let toggleOverConstraint = document.getElementById("toggleOverConstraint").checked;

    // Clear the graph
    cy.$().remove()
    pattern.userIMLEdits = {}

    // Get the hint rows map
    let hintRowsMap = createHintRowsMap()

    // Only run this on approved classes
    let rows = getApprovedRows()
    console.log("rows")
    print(rows)

    // Nodes
    for(let i = 0; i < rows.length; i++){
        let {uid} = rows[i]
        addNodeToGraph(uid)
    }

    // Edges
    // Do this by looping over each connection and finding valid connections

    // Each possible row to row connection
    for(let i = 0; i < rows.length; i++){
        let pkg_i = rows[i]
        let row_i = pkg_i.tableRow
        let uid_i = pkg_i.uid
        let hintRows_i = hintRowsMap[uid_i]
        // let tableName_i = getTableNameFromPatternWithUID(uid_i)
        // let header_i = tableHeaders[tableName_i]
        for(let j = i+1; j < rows.length; j++){
            let pkg_j = rows[j]
            let row_j = pkg_j.tableRow
            let uid_j = pkg_j.uid
            let hintRows_j = hintRowsMap[uid_j]
            // let tableName_j = getTableNameFromPatternWithUID(uid_j)
            // let header_j = tableHeaders[tableName_j]
            
            // Get possible edges between these rows, if they exist then draw them
            let [possibleEdges, possibleEdges_over] = determinePossibleEdges(row_i, row_j)
            if(possibleEdges.length > 0){
                // *********************
                let edge = addEdgeToGraph(uid_i, uid_j, possibleEdges, constraints_over=possibleEdges_over)

                // Because both nodes could have hint rows we have to compare each hint row to all other possible rows in the other node
                
                // If we are overconstraining, dont remove the problems
                if(toggleOverConstraint) markProblemHintRows(hintRows_i, hintRows_j, possibleEdges_over, edge)

                // Determine if there are any hint rows that do not meet the constraints
                // *** This should be redundant now
                else removeProblemConstraints(hintRows_i, hintRows_j, possibleEdges, edge)
            }
        }   
    }


    // Classify the edges so the IML can use that info
    classifyGraphConstraints()

    cy.nodes().forEach(markIssuesOnNode)
    runLayout()
}

// Populate the cytoscape graph from the edge table data
function graphFromEdgeTable(){
    
    // Clear the problem class markers at the start
    cy.nodes().removeClass('problem')
    
    // Needed for marking nodes/edges with problematic hint rows
    let hintRowsMap = createHintRowsMap()

    // Read current edge table
    let patternName = patternNames[patternIdx]
    let edgeTable = edgeTablesMap[patternName]
    let uids = edgeTable.header

    // Clear the graph
    cy.$().remove()

    // Populate the nodes
    for(let uid of uids){
        addNodeToGraph(uid)
    }

    // Populate the edges
    for(let i in uids){
        let uid_i = uids[i]
        for(let j in uids){
            let uid_j = uids[j]
            let constraints = constraintsFromEdgeTableStr(edgeTable[uid_i][uid_j])
            if(constraints.length > 0){
                let edge = addEdgeToGraph(uid_i, uid_j, constraints)

                markProblemHintRows(hintRowsMap[uid_i], hintRowsMap[uid_j], constraints, edge)
                // removeProblemConstraints(hintRowsMap[uid_i], hintRowsMap[uid_j], constraints, edge)
            }
        }   
    }

    // If we have no IML info on file determine graph constraints
    // Classify the edges so the IML can use that info
    print(pattern.userIMLEdits)
    classifyGraphConstraints()
    print(pattern.userIMLEdits)
    
    cy.nodes().forEach(markIssuesOnNode)
    runLayout()
}

/*
 * HTML Populate/Style
 */ 

// Determine the number of hint rows for this row in the current pattern
function getHintRowCount(uid){
    for(let i = 0; i < pattern.length; i++){
        if(pattern['UID'][i] == uid) return pattern['hintRowUUIDs'][i].length
    }
    return NaN
}

// Get a color for the edge based on its rows
function getColorForEdge(uid_from, uid_to, flag){

    let backgroundColor = 'white'
    if(flag) {
        let role_from = getRoleShortForUIDInPattern(uid_from)
        let role_to = getRoleShortForUIDInPattern(uid_to)

        // prioritize the row with more hints in it for color coding
        let count_from = getHintRowCount(uid_from)
        let count_to = getHintRowCount(uid_to)
        if(count_from >= count_to) role = role_from
        else role = role_to

        switch(role){
            case 'G':
                backgroundColor = '#fdd0a2' // ["#fff5eb","#fee6ce","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#a63603","#7f2704"]
                break
            case 'SW':
                backgroundColor = '#c6dbef' // ["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#08519c","#08306b"]
                break
            default:
                backgroundColor = '#c7e9c0' // ["#f7fcf5","#e5f5e0","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#006d2c","#00441b"]
                break
        }
    }
    else backgroundColor = '#FC9993' //["#fff5f0","#fee0d2","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#a50f15","#67000d"]

    return backgroundColor
}

// Checks to see if a given UUID is found in the hints for any row.  If so, it returns a list of UUIDS that contain that row as a hint. 
function getRowsThisRowIsAHintRowFor(uuid) {
    let out = [];

    // Check to see if hintUUID values are popuated -- if not, then nothing to search. 
    if (!pattern.hasOwnProperty('hintRowUUIDs')) {        
        return [];
    }

    for (let i=0; i<pattern['UID'].length; i++) {
        let masterUUID = pattern['UID'][i]
        let hintUUIDs = pattern["hintRowUUIDs"][i];

        for (let j=0; j<hintUUIDs.length; j++) {
            if (hintUUIDs[j] == uuid) {
                out.push(masterUUID);
            }
        }
    }

    return out;
}

// Checks to see if a given UUID is found in the hints for any row.  If so, it returns a list of UUIDS that contain that row as a hint. 
function isRowAHintInAnotherRow(uid) {
    for(let i = 0; i < pattern.length; i++){
        if(pattern["hintRowUUIDs"][i].includes(uid)) return true
    }
    return false
}

// Return error status for the row
function errorCheckRow(uid){
    let hasMisMatchedTables = false
    let isMissingHintRows = false
    let isFailingConstraints = false
    let hasPossibleMisplacedHintRow = false

    // Skip these rows
    if(!isRowAHintInAnotherRow(uid)){

        let row = getTableRow(uid)

        // Read in important variables
        let i = pattern['UID'].indexOf(uid)
        let hintRows = pattern['hintRowUUIDs'][i]
        let rating = pattern['RATING'][i]
    
        // If the row is swappable
        if(isSwappableRow(rating)){
    
            // missing hint row
            if(hintRows.length == 0) isMissingHintRows = true
            // Check that all hint rows are in the same table
            else {
                let nodeTable = row.tablename
                for(let hintRowUID of hintRows){
                    let hintRow = getTableRow(hintRowUID)
                    let rowTable = hintRow.tablename
                    if(nodeTable != rowTable) hasMisMatchedTables = true
                    else nodeTable = rowTable
                }
            }
        }
    
        // Read the cytoscape node for this row and see if it was flaged as a problem
        // Note, only seed rows are present in cytoview
        if(isSeedRow(rating)){
            let node = cy.nodes(`#${uid}`)[0]
            if(typeof node == 'object'){
                isFailingConstraints = node.hasClass('problem')
                hasPossibleMisplacedHintRow = node.hasClass('warning')
            } else{
                console.log(`%cERROR! "${uid}" is a seed row but isn't in the graph.`, 'color:red')
            }
        }
    }

    return {
        hasMisMatchedTables:hasMisMatchedTables, 
        isMissingHintRows:isMissingHintRows,
        isFailingConstraints:isFailingConstraints,
        hasPossibleMisplacedHintRow: hasPossibleMisplacedHintRow,
    }
}

// Toggle warnings based on errors
function markWarningsOnRow(rowEle, uid){

    let {hasMisMatchedTables, isMissingHintRows, isFailingConstraints, hasPossibleMisplacedHintRow} = errorCheckRow(uid)
    // console.log('1')
    // console.log({hasMisMatchedTables:hasMisMatchedTables, isMissingHintRows:isMissingHintRows, isFailingConstraints:isFailingConstraints, hasPossibleMisplacedHintRow:hasPossibleMisplacedHintRow})

    // Read the row text locally
    let text = rowEle.innerHTML

    // If a warning is already here remove it 
    // Note: order of removal must invert the order appended (below)
    let hasFailingConstraintsWarningAppended = text.includes(FAILING_CONSTRAINTS_WARNING)
    let hasMisMatchWarningAppended = text.includes(MISMATCHED_TABLES_WARNING)
    let hasMissingWarningAppended = text.includes(MISSING_HINTROW_WARNING)
    let hasMisplaceWarningAppended = text.includes(POSSIBLE_MISPLACED_HINTROW_WARNING)
    if(hasFailingConstraintsWarningAppended) text = text.substring(0, text.length - FAILING_CONSTRAINTS_WARNING.length)
    if(hasMisMatchWarningAppended) text = text.substring(0, text.length - MISMATCHED_TABLES_WARNING.length)
    if(hasMissingWarningAppended) text = text.substring(0, text.length - MISSING_HINTROW_WARNING.length)
    if(hasMisplaceWarningAppended) text = text.substring(0, text.length - POSSIBLE_MISPLACED_HINTROW_WARNING.length)

    // Apply the appropriate warning message
    if(hasPossibleMisplacedHintRow) text += POSSIBLE_MISPLACED_HINTROW_WARNING
    if(isMissingHintRows) text += MISSING_HINTROW_WARNING
    if(hasMisMatchedTables) text += MISMATCHED_TABLES_WARNING
    if(isFailingConstraints) text += FAILING_CONSTRAINTS_WARNING
 
    rowEle.innerHTML = text
}

// Return the html string for a row in the table thing
// requires an idx for the radio buttons to work
function mkRatingRowHTML(rowRating, internalIdx, rowIdxForHuman, patIdx, hintRows, listedAsHintFor="") {
    let rowText = rowRating.row.text;
    let uid = rowRating.row.uuid;
    let count = rowRating.count; 
    let isOptional = rowRating.optional;
    let rating = rowRating.rating
    let backgroundColor = getRatingHTMLColorCode(rating, rowIdxForHuman);

    // Error checking
    // If row is swappable check that there are hint rows or that those rows are from the same table
    let {hasMisMatchedTables, isMissingHintRows, isFailingConstraints, hasPossibleMisplacedHintRow} = errorCheckRow(uid)
    // console.log('2')
    // console.log({hasMisMatchedTables:hasMisMatchedTables, isMissingHintRows:isMissingHintRows, isFailingConstraints:isFailingConstraints, hasPossibleMisplacedHintRow:hasPossibleMisplacedHintRow})
    
    if (count === 'undefined') count = 0;

    // Get pivot words
    let hintWords = pattern['hintWords'][patIdx]
    let pivotWordsText = ""
    if(typeof hintWords !== 'undefined') {
        pivotWordsText = hintWords.join(', ')
    }
    
    // Create string with hint row text
    var hintRowText = ""
    if (pivotWordsText.length > 0) hintRowText = `<br><font color=gray>Pivot words: ${pivotWordsText}</font>` 

    for (let i=0; i<hintRows.length; i++) {
        let hintUID = hintRows[i]
        let row = getTableRow(hintUID)
        let cleanText = getTextFromTableRow(row)
        hintRowText += "<br><font color=grey>" + cleanText + "</font>";
         
        // Limit to displaying a maximum of 4 possible substitions
        if ((i >= 3) && (hintRows.length > i+1)) {
            hintRowText += "<br><font color=grey> ... </font>";
            break;
        }
    }

    // Get row notes
    let rowNotes = pattern['rowNotes'][patIdx]
    if(rowNotes){
        if (rowNotes.length > 0) hintRowText += "<br><font color=grey><i>Notes: " + rowNotes + "</i></font>"
    } else{
        pattern['rowNotes'][patIdx] = ''
        rowNotes = ''
        if (rowNotes.length > 0) hintRowText += "<br><font color=grey><i>Notes: </i></font>"
    }

    // Append warning if row is missing hint rows
    if(hasPossibleMisplacedHintRow) hintRowText += POSSIBLE_MISPLACED_HINTROW_WARNING
    if(isMissingHintRows) hintRowText += MISSING_HINTROW_WARNING
    if(hasMisMatchedTables) hintRowText += MISMATCHED_TABLES_WARNING
    if(isFailingConstraints) hintRowText += FAILING_CONSTRAINTS_WARNING


    // Check to see if this row is listed as a hint row for any other rows
    let classStr = ""
    if (listedAsHintFor.length > 0) {
        //##
        hintRowText += "<br><font color=red>NOTE: This row is listed as a substitution for: " + listedAsHintFor + "</font>" 
        backgroundColor = "#DD0000";
        classStr = "class=\"" + getRatingHTMLStripeCode(rowRating.rating) + "\"";
    }

    // Check to see if this row is listed as optional
    // console.log("isOptional: " + isOptional + "  text: " + rowText)
    if (isOptional == true) {
        hintRowText += OPTIONAL_ROW_STR
    }
    
    var ratioIconHTML = ""
    var ratioCentral = -1
    let freqs = rowRoleFreqs[uid]
    if (freqs) {
        ratioCentral = 0.0
        let keys = Object.keys(freqs)
        for(key of keys){
            if(freqs.hasOwnProperty(key)){
                ratioCentral += parseFloat(freqs[key])
            }
        }

        // Get ratio and round
        ratioCentral = parseFloat(freqs['CENTRAL']) / ratioCentral
        ratioCentral = (Math.round(ratioCentral * 10) / 10)
        if (ratioCentral === 'undefined') {
            ratioCentral = -1;
        }
        // toString doesnt respect the radix
        if(ratioCentral === 0.0){
            ratioCentral = '0.0'
        } else if(ratioCentral === 1.0){
            ratioCentral = '1.0'
        } else{
            ratioCentral = ratioCentral.toString()
        }

        // Use icon for central ratio         
        if (ratioCentral >= 0.66) {
            ratioIconHTML = "<i class=\"fas fa-circle\"></i>"
        } else if (ratioCentral >= 0.33) {
            ratioIconHTML = "<i class=\"fas fa-adjust\"></i>"
        } else {
            ratioIconHTML = "<i class=\"far fa-circle\"></i>"
        }

    }

    if (ratioCentral == -1) {
        ratioCentral = "";
        ratioIconHTML = "n/a";
        // console.log(`N/a ratio "${uid}"`)        
    }

    return `
    <tr id="row_${internalIdx}" bgcolor="${backgroundColor}" ${classStr}>
        <td>
            ${rowIdxForHuman}
        </td>
        <td class="buttonCol">
            <button class="button central" onclick="voteCallback('${internalIdx}', '${RATING_CENTRAL}')" data-tooltip="Central"> <i class="fas fa-plus-square"></i> </button>
            <button class="button centralsw" onclick="voteCallback('${internalIdx}', '${RATING_CENTRALSW}')" data-tooltip="Central (Switchable)"> <i class="far fa-plus-square"></i> </button>
            <button class="button grounding" onclick="voteCallback('${internalIdx}', '${RATING_GROUNDING}')" data-tooltip="Grounding"> <i class="fas fa-plus"></i> </button>            
            <button class="button lexglue" onclick="voteCallback('${internalIdx}', '${RATING_LEXGLUE}')" data-tooltip="Lexical Glue"> <i class="fas fa-share-alt-square"></i> </button>            
            <button class="button maybe" onclick="voteCallback('${internalIdx}', '${RATING_MAYBE}')" data-tooltip="Maybe"> <i class="far fa-question-circle"></i> </button>            
            <button class="button optional" onclick="voteCallbackOptional('${internalIdx}')" data-tooltip="Optional"> <i class="far fa-square"></i> </button>
            <button class="button bad" onclick="voteCallback('${internalIdx}', '${RATING_BAD}')" data-tooltip="Bad/Not Relevant"> <i class="fas fa-minus-square"></i> </button>
        </td>
        <td class="count"><center>${count}</center></td>
        <td class="ratio"><center> ${ratioIconHTML} ${ratioCentral} </center></td>
        <td id="row_${internalIdx}_text" class="align-left">${rowText} [${rowRating.row.tableName}] ${hintRowText}</td>
    </tr>
    `
}

function getRatingHTMLColorCode(rating, index) {
    if (rating == RATING_CENTRAL) {
        return "#ADE190";                   // D3 Green
    } else if (rating == RATING_CENTRALSW) {
        return "#aec7e8";                   // D3 Blue
    } else if (rating == RATING_GROUNDING) {
        return "#F9BF76";                   // D3 Orange
    } else if (rating == RATING_LEXGLUE) {
        return "#cab2d6";
    } else if (rating == RATING_MAYBE) {
        return "#dadaeb";
    } else if (rating == RATING_BAD) {
        return "#FC9993";                   // D3 Red
    } else if (rating == RATING_UNRATED) {
        if (index % 2 == 0) {
            return "#FFFFFF";                   // Unrated  (alternating colors)
        } else {
            return "#EEEEEE";                   // Unrated  (alternating colors)
        }
    } 

    return "#E11918";                       // This should never happen (D3 bright red)
}

function getRatingHTMLStripeCode(rating) {
    if (rating == RATING_CENTRAL) {
        return "stripe-central";                   // D3 Green
    } else if (rating == RATING_CENTRALSW) {
        return "stripe-centralsw";                 // D3 Blue
    } else if (rating == RATING_GROUNDING) {
        return "stripe-grounding";                 // D3 Orange
    } else if (rating == RATING_MAYBE) {
        return "stripe-maybe";
    } else if (rating == RATING_BAD) {
        return "stripe-bad";                       // D3 Red
    } else if (rating == RATING_UNRATED) {
        return "stripe-greyedout"
    } 

    return "";                       // This should never happen
}

// Focus on the row selected: currently uses a blue border
function focusRowSelected(){
    
    let focusRow = null
    let rows = $('table#votingTable tr')
    for(row of rows){
        let id = row.id
        if(id === currentRow){
            $(row).css('border', '2px solid #3787c0')
            focusRow = row
        }
        else{
            $(row).css('border', '1px solid white')
        }
    }

    // Scrollie Pollie Ollie
    // Dynamically responds to table height
    // scrolls when we hit the edges
    if(focusRow !== null){
        let scrollTop = $("#votingTable tbody").scrollTop()
        let height = $("#votingTable tbody").height()
        let delta = focusRow.offsetTop - scrollTop
        if(height > 50){    // On page load it will have weird values for height
            if(height - 22 < delta){
                $("#votingTable tbody").scrollTop(scrollTop + (delta - height+22))
            }
            else if(delta < 65){
                $("#votingTable tbody").scrollTop(scrollTop - (65-delta))
            }
        }
    }
    
}

// Color a given row based on rating
function colorRowByRating(rowRefStr, rating, uuid) {
    // Get cells of the row for coloring    
    let children = $(rowRefStr).children('td')
    
    // Handle the warning messages
    let rowEle = children.last()[0]
    markWarningsOnRow(rowEle, uuid)

    children.css('background-color', getRatingHTMLColorCode(rating))

    // Check to see if this row is listed as a hint row for any other rows -- if so, highlight
    if(isRowAHintInAnotherRow(uuid)) {
        let possibleClasses = ["stripe-greyedout", "stripe-central", "stripe-centralsw", "stripe-grounding", "stripe-maybe", "stripe-bad"]
        for (let i=0; i<possibleClasses.length; i++) {
            children.removeClass(possibleClasses[i]);
        }        
        children.addClass(getRatingHTMLStripeCode(rating))
    }
}

// Handles the seed box html
function populateHTMLSeedBox() {
    var centralRows = [];
    var centralSWRows = [];
    var groundingRows = [];

    // Step 1: Collect central and grounding rows
    for (var i=0; i<rowRatingsForCurrentPattern.length; i++){
        var rowRating = rowRatingsForCurrentPattern[i];

        if (rowRating.rating == RATING_CENTRAL) {
            centralRows.push(rowRating);            
        } else if (rowRating.rating == RATING_CENTRALSW) {
            centralSWRows.push(rowRating);
        } else if (rowRating.rating == RATING_GROUNDING) {
            groundingRows.push(rowRating);
        }
    }

    // Step 2: Sort by frequency
    // console.log(centralRows)
    var sortedCentralRows = centralRows.sort((a, b) => (a.count < b.count ? 1 : -1))
    var sortedCentralSWRows = centralSWRows.sort((a, b) => (a.count < b.count ? 1 : -1))
    var sortedGroundingRows = groundingRows.sort((a, b) => (a.count < b.count ? 1 : -1))

    // Step 3: Populate HTML box
    var htmlOutStr = ""
    for (var i=0; i<sortedCentralRows.length; i++) {
        var rowRating = sortedCentralRows[i]
        htmlOutStr += `<tr> <td>${rowRating.rating}</td><td>${rowRating.count}</td><td class="align-left">${rowRating.row.text} [${rowRating.row.tableName}]</td> </tr>`
    }
    for (var i=0; i<sortedCentralSWRows.length; i++) {
        var rowRating = sortedCentralSWRows[i]
        htmlOutStr += `<tr> <td>${rowRating.rating}</td><td>${rowRating.count}</td><td class="align-left">${rowRating.row.text} [${rowRating.row.tableName}]</td> </tr>`
    }
    for (var i=0; i<sortedGroundingRows.length; i++) {
        var rowRating = sortedGroundingRows[i]
        htmlOutStr += `<tr> <td>${rowRating.rating}</td><td>${rowRating.count}</td><td class="align-left">${rowRating.row.text} [${rowRating.row.tableName}]</td> </tr>`
    }

    $('#seedRowTable').html(`
    <thead>
    <tr>
        <th>Role</th><th>Count</th><th>Row Text</th>
    </tr>
    </thead>
    <tbody>
    ` + htmlOutStr + "<tbody>")
}

// Hndles the rating box html
function populateHTMLRatingBox() {
    // Sort by selected option
    let sortedRows = []

    let selectedSort = getSelectedOptionValue('sortSelector')

    switch(selectedSort){
        case "TermFreq":
            let seedLemmas = getSeedLemmas()
            sortedRows = rowRatingsForCurrentPattern.sort(function(a, b){
        
                if(typeof a.row.text === 'undefined') return 1
                if(typeof b.row.text === 'undefined') return -1
        
                let wordsA = getTextStopwordsRemoved(a.row.text);   // a.row.text.split(' ')
                let wordsB = getTextStopwordsRemoved(b.row.text);   // b.row.text.split(' ')
                    
                let numWordsOvlA = 0
                let numWordsOvlB = 0
        
                // TODO: Remove stop words?
                for(word of wordsA){
                    if(seedLemmas.has(lemmatize(word))) numWordsOvlA ++
                }
                for(word of wordsB){
                    if(seedLemmas.has(lemmatize(word))) numWordsOvlB ++
                }
        
                // First, sort by number of overlapping words
                if (numWordsOvlB > numWordsOvlA) return 1;
                if (numWordsOvlA > numWordsOvlB) return -1;

                // If the number of overlapping words is the same, sort by text
                if (b.row.text < a.row.text) return 1;
                if (a.row.text < b.row.text) return -1;

                // If we reach here, the two rows are the same. 
                return 0;                                
            })            
            break
        case "Count":
            sortedRows = rowRatingsForCurrentPattern.sort(function(a, b) {
                let aCount = a.count
                let bCount = b.count
                if (typeof aCount === 'undefined') aCount = -1 // Safety
                if (typeof bCount === 'undefined') bCount = -1 // Safety

                // First, sort by count
                if (bCount > aCount) return 1;
                if (aCount > bCount) return -1;

                // If the count is the same, sort by text                
                if (b.row.text < a.row.text) return 1;
                if (a.row.text < b.row.text) return -1;

                // If we reach here, the two rows are the same. 
                return 0;                                
            })            
            break
        default:
            sortedRows = rowRatingsForCurrentPattern.sort()            
            break
    }    

    // Step 3: Populate HTML box
    let numPossibleMissplacedHintRows = 0
    let numRowsMissingHintRows = 0
    let numRowsMismatchedTables = 0
    let numRowsInvalidHintRows = 0
    var htmlOutStr = ""
    var htmlOutStrBad = ""      // Bad rows go to the bottom
    var htmlOutStrStriped = ""  // Place the striped (ie rows that already appear in another rows hints) on the bottom
    for (let i=0; i<sortedRows.length; i++) {
        var rowRating = sortedRows[i]        
        var internalIdx = rowRatingsForCurrentPattern.indexOf(rowRating); 
        let uid = rowRating.row.uuid
        let rating = rowRating.rating
        
        let idx = pattern['UID'].indexOf(uid)
        let hintRows = []
        if (idx >= 0) hintRows = pattern['hintRowUUIDs'][idx];
        // console.log("Hint Rows for " + rowRating.row.text + " : ")
        // console.log(rating)

        // Append to the table HTML
        var listedAsHintFor = getRowsThisRowIsAHintRowFor(uid)
        if (listedAsHintFor.length > 0) htmlOutStrStriped += mkRatingRowHTML(rowRating, internalIdx, i+1, idx, hintRows, listedAsHintFor);
        else if (rating === RATING_BAD) htmlOutStrBad += mkRatingRowHTML(rowRating, internalIdx, i+1, idx, hintRows, listedAsHintFor);
        else htmlOutStr += mkRatingRowHTML(rowRating, internalIdx, i+1, idx, hintRows, listedAsHintFor);

        let {hasMisMatchedTables, isMissingHintRows, isFailingConstraints, hasPossibleMisplacedHintRow} = errorCheckRow(uid)
        // console.log('0')
        // console.log({hasMisMatchedTables:hasMisMatchedTables, isMissingHintRows:isMissingHintRows, isFailingConstraints:isFailingConstraints, hasPossibleMisplacedHintRow:hasPossibleMisplacedHintRow})
        if(hasPossibleMisplacedHintRow) numPossibleMissplacedHintRows++
        if(isMissingHintRows) numRowsMissingHintRows++
        if(hasMisMatchedTables) numRowsMismatchedTables++
        if(isFailingConstraints) numRowsInvalidHintRows++
    }

    document.getElementById('numPossibleMissplacedHintRows').innerHTML = numPossibleMissplacedHintRows
    document.getElementById('numRowsMissingHintRows').innerHTML = numRowsMissingHintRows
    document.getElementById('numRowsMismatchedTables').innerHTML = numRowsMismatchedTables
    document.getElementById('numRowsInvalidHintRows').innerHTML = numRowsInvalidHintRows

    // Append the striped rows at the bottom
    htmlOutStr += htmlOutStrBad
    htmlOutStr += htmlOutStrStriped

    // Set the voting table html
    $('#votingTable').html(`
        <thead>
            <th>#</th><th>Buttons</th><th>Count</th><th>Ratio Central</th><th>Row Text</th>
        </thead>
        <tbody>
    ` + htmlOutStr + "</tbody>")    

    // Add onclick events to the rows
    // This selects the current focused row
    $("table#votingTable tr > *").click(function(){

        // Make sure that buttons dont trigger this
        if(!this.classList.contains('buttonCol')){
            currentRow = this.parentElement.id
            focusRowSelected()
        }
    })

    $("table#votingTable tr > *").dblclick(function(){

        // Make sure that buttons dont trigger this
        if(!this.classList.contains('buttonCol')){
            currentRow = this.parentElement.id
            focusRowSelected()
            openHintAddingWindow();
        }
    })    

    // Call it once to default
    focusRowSelected()
}

// Callback for when the pattern selector is changed
function populatePatternHTML(){

    // Step 1: Populate seed box
    populateHTMLSeedBox();

    // Step 2: Populate rating box
    populateHTMLRatingBox();
    
    // Step 3: Initialize statistics for user
    updateStatistics()

    // Step X: Size the table height dynamically
    sizeVotingTable()
}

// 
function populatePatternSeletor(){

    // Get queries from pattern query
    let queries = patternQuery.split(' ')
    // console.log(queries)
            
    // Populate the selector options (string)
    let patternOptions = ""
    for(i in patternNames){
        let patternName = patternNames[i]
        let localPattern = patterns[i]
        let localPatternIsRedundant = localPattern.isRedundant
        let localPatternIsBad = localPattern.isBad
        let localPatternIsUncertain = localPattern.isUncertain
        let localPatternIsGood = localPattern.isGood
        let localPatternIsDone = localPattern.isDone
        let localPatternIsUnmarked = !localPatternIsRedundant && !localPatternIsBad && !localPatternIsUncertain && !localPatternIsGood && !localPatternIsDone

        // Check for matches
        let matchCount = 0
        for(word of queries){
            if(patternName.toLowerCase().indexOf(word.toLowerCase()) >= 0) matchCount++
        }

        // If all queries match then add to list
        // Filter out the patterns that we arent showing
        if(matchCount === queries.length) {
                 if(localPatternIsRedundant && showRedundant)   patternOptions += `<option value="${patternName}">(r) ${patternName}</option>\n`
            else if(localPatternIsBad && showBad)               patternOptions += `<option value="${patternName}">(b) ${patternName}</option>\n`
            else if(localPatternIsUncertain && showUncertain)   patternOptions += `<option value="${patternName}">(?) ${patternName}</option>\n`
            else if(localPatternIsGood && showGood)             patternOptions += `<option value="${patternName}">(g) ${patternName}</option>\n`
            else if(localPatternIsDone && showDone)             patternOptions += `<option value="${patternName}">(d) ${patternName}</option>\n`
            else if(localPatternIsUnmarked && showUnmarked)     patternOptions += `<option value="${patternName}">    ${patternName}</option>\n`
        }
    }

    // Post it in the HTML
    $('#patternSelector').html(patternOptions)

    // Select first option
    let firstSelection = $(`#patternSelector option`)[0].value
    selectByValue('patternSelector', firstSelection)
}

// Fill up the edge voting table
function populateEdgeTableHTML(){

    // UID of base row
    let uid_0 = getCurrentlySelectedUID()

    // Get node object for this uid
    let node_0 = cy.nodes(`#${uid_0}`)[0]
    if(typeof node_0 == 'object'){

        // Get all edges connected to this node so we can get the constraints
        let edges = cy.edges(function(edge){
            return edge.data('source') == uid_0 || edge.data('target') == uid_0
        })

        // Store all the constraints as JSON objects in this list
        let constraintsList = []
        for(let i = 0; i < edges.length; i++){
            let edge = edges[i]
            let uid_from = edge.data('source')
            let uid_to = edge.data('target')

            let constraints = edge.data('constraints')
            for (let {colIdx_from, colIdx_to, on} of constraints){

                let numHintRows = getHintRowCount(uid_from) + getHintRowCount(uid_to)

                constraintsList.push({
                    uid_from: uid_from,
                    uid_to: uid_to,
                    col_from: colIdx_from,
                    col_to: colIdx_to,
                    on: on,
                    numHintRows: numHintRows,
                })
            }
        }

        // Sort the edges by num of hint rows in its rows
        constraintsList.sort((a, b) => b.numHintRows - a.numHintRows)

        // Populate the table with the edges
        let strOut = ''
        for(let i in constraintsList){
            let {uid_from, uid_to, col_from, col_to, on, numHintRows} = constraintsList[i]

            // Read the text from the tablestore
            // order them so that the main row is always first
            let text_from = tableMap[uid_from]
            let cells_from = text_from.split(' | ')
            let text_to = tableMap[uid_to]
            let cells_to = text_to.split(' | ')
    
            // Determine coloring for the table
            let color = 'black'
            if(numHintRows == 0) color = 'grey'
            let backgroundColor = getColorForEdge(uid_from, uid_to, on)
    
            // Add the first two columns onto the html string
            strOut += `<tr id='edge_row_${i}' style="color:${color}; background-color:${backgroundColor};"> 
                        <td> 
                            <button class="yButton hoverable" onclick='markEdge("${uid_from}", "${uid_to}", "${col_from}", "${col_to}", ${i}, true)'> + </button> 
                            <button class="nButton hoverable" onclick='markEdge("${uid_from}", "${uid_to}", "${col_from}", "${col_to}", ${i}, false)'> - </button> 
                        </td>
                        <td>${numHintRows}</td>
                        <td class='align-left'>`
    
            // Populate the strings for the text col
            let sourceStr = ''
            let targetStr = ''

            // Source row
            sourceStr += `[${getTableNameFromText(cells_from[cells_from.length-1])}] `
            for(let n = 0; n < cells_from.length-1; n++){
                let cell_from = cells_from[n]
    
                if(col_from == n) sourceStr += `&nbsp;<b style='color:blue'>[${cell_from}]</b>`
                else{
                    if(cell_from.length > 0) sourceStr += ` ${cell_from}`
                }
            }
    
            sourceStr += '\n<br>\n'
    
            // Source Row: Hint Rows
            let hint_uids = getHintRows(uid_from)
            if(hint_uids.length > 0){
                sourceStr += "<p style='color:grey'>\n"
                for(hint_uid of hint_uids){
                    let cells = tableMap[hint_uid].split(' | ')
                    for(let n = 0; n < cells.length-1; n++){
                        let cell = cells[n]
            
                        if(col_from == n) sourceStr += `<b>[${cell}]</b>&nbsp;`
                        else{
                            if(cell.length > 0) sourceStr += `${cell} `
                        }
                    }
                    sourceStr += '\n<br>\n'
                }
                sourceStr += "</p><br>\n"
            } else{
                sourceStr += "<br>\n"
            }
            
            // Target row
            targetStr += `[${getTableNameFromText(cells_to[cells_to.length-1])}] `
            for(let n = 0; n < cells_to.length-1; n++){
                let cell_to = cells_to[n]
    
                if(col_to == n) targetStr += `&nbsp;<b style='color:blue'>[${cell_to}]</b>`
                else {
                    if(cell_to.length > 0) targetStr += ` ${cell_to}`
                }
            }
    
            // Target Row: Hint Rows
            hint_uids = getHintRows(uid_to)
            if(hint_uids.length > 0){
                targetStr += "<p style='color:grey'>\n"
                for(hint_uid of hint_uids){
                    let cells = tableMap[hint_uid].split(' | ')
                    for(let n = 0; n < cells.length-1; n++){
                        let cell = cells[n]
            
                        if(col_to == n) targetStr += `<b>[${cell}]</b>&nbsp;`
                        else{
                            if(cell.length > 0) targetStr += `${cell} `
                        }
                    }
                    targetStr += '\n<br>\n'
                }
                targetStr += "</p>\n"
            }

            // Ensure the base UID is appended first
            if(uid_from != uid_0){
                strOut+= sourceStr
                strOut+= targetStr
            } else{
                strOut+= targetStr
                strOut+= sourceStr
            }
    
            // Close this table row
            strOut += '</td>\n</tr>\n'
        }
    
        $('#edgeVoterTable').html(`
        <thead>
        <tr>
            <th> +/- </th><th> # </th><th> Edge </th>
        </tr>
        </thead>
        <tbody>
        ` + strOut + "</tbody>")
    }
}

// Function to handle setting up the user IML editing table
function populateUserIMLEditingTable(){

    // Retrieve current user IML edits
    let userIMLEdits = pattern.userIMLEdits
    let lemmasToVarNameMap = userIMLEdits.lemmasToVarNameMap

    // Loop through each hash entry and generate a row for each
    let strOut = "<table id='imlVarViewTable'>\n"

    strOut += "<thead>\n"
    strOut +=    "<tr> <th>Expression</th> <th>Text</th> </tr>\n"
    strOut += "</thead>\n\n"

    strOut +=    "<tbody>\n"

    // Determine the alts for vars, while simultaniously determining the lemmas used
    let varAlts = {}
    let lemmasUsed = new Set()

    // Get an array that can loop through the map.
    let userIMLEditsItter = Object.entries(userIMLEdits).filter((a) => isNumber(a[0]))
    for(let [_, {expression, edgeText}] of userIMLEditsItter){
        
        let expressions = expression.split("+")
        for(let expression_i of expressions){
            // Lemma
            if(expression_i.substring(0, 1) == '"'){
                lemmasUsed.add(expression_i)
            }
            else{
                varAlts[expression_i] = edgeText
            }
        }
    }
    console.log(varAlts)

    let varAltsItter = Object.entries(varAlts).sort((a,b) => a[0].localeCompare(b[0]))
    console.log(varAltsItter)
    for(let [varName, edgeText] of varAltsItter) {
        let edgeTextHTML = edgeText.replace(/\n/g, "<br>")

        let htmlFriendlyExpression = varName.replace(/</g, "&lt;")
        htmlFriendlyExpression = htmlFriendlyExpression.replace(/>/g, "&gt;")

        strOut += `<td>${htmlFriendlyExpression}</td> <td>${edgeTextHTML}</td> </tr>\n`
    }

    lemmasUsed = [...lemmasUsed].sort()
    for(let lemma of lemmasUsed){
        strOut += `<td>${lemma}</td> <td>${lemma.substring(1, lemma.length-1)}</td> </tr>\n`
    }
    
    strOut +=    "</tbody>\n"
    strOut +=    "</table>\n"

    document.getElementById("imlVarView").innerHTML = strOut
}

// Fills the viewer with an IML export of the current patern
function populateIMLViewer(){
    
    let iml_auto = createIMLFromPattern()

    let iml_manual = pattern.iml
    let isManualIML = pattern.isManualIML

    if(iml_manual.length == 0) {
        pattern.iml = iml_auto
        iml_manual = iml_auto
    }

    // If manual changes
    if(isManualIML){

        // If local file deviates from what we calculate then there 
        // have been chagnes to the pattern since the IML was saved
        let manualEditsStr = "// NOTE: This IML has been edited from it's original state"
        if (!iml_manual.startsWith(manualEditsStr)) {
            iml_manual = manualEditsStr + "\n\n" + iml_manual
        }   

        editorIML.setValue(iml_manual)
    }
    else{
        editorIML.setValue(iml_manual)
    }

    dataInEditor = true

    changesSinceLastIMLPopulate = false
}

// Sets the title as needed
function setPatternName(name){
    $('#patternName').html(name)
}

// Resize the main window
function sizeVotingTable(){

    //document.getElementById("votingTableDiv").style.height = (tableHeight + delta - notesHeight) + "px";
    let divHeight = $( window ).height() - 568
    let divHeightPx = divHeight + "px"
    document.getElementById("votingTableDiv").style.height = divHeightPx
    document.getElementById("votingTable").style.height = divHeightPx
    // document.getElementById("graphDiv").style.height = (divHeight - 8) + "px"
    $("#votingTable tbody")[0].style.height = (divHeight - 56) + 'px'
}

// Mark constraint as good or bad
function markEdge(uid_from, uid_to, col_from, col_to, votingTableIdx, flag){

    // get the edge associated with these two uids
    let edge = cy.edges(function(edge){
        return edge.data('source') == uid_from && edge.data('target') == uid_to
    })[0]

    if(typeof edge == 'object'){
           
        // Created a string for the constraint with its vote
        let constraintFullStr = (flag)? `${col_from}:${col_to}:Y` : `${col_from}:${col_to}:N`;

        // Find the constraint being changed by this callback and change it
        let constraints = edge.data('constraints')
        for(let i in constraints){
            let {col_from, col_to} = constraints[i]

            // This is the constraint that changed
            if(col_from == col_from_i && col_to == col_to_i) {
                constraints[i].rating = flag
            }
        }

        // Store the constraint back in the edge obj
        edge.data('constraints', constraints)

        // update the edge label
        let text_from = tableMap[uid_from]
        let text_to = tableMap[uid_to]
        let label = createEdgeLabel(constraints, text_from, text_to)
        edge.data('label', label)

        // set the edge as visible of invisible based on label
        if(label.length == 0) edge.style('visibility', 'hidden')
        else edge.style('visibility', 'visible')

        // Color the row
        // TODO: This doesn't work, somehow it is ignoring this value when coloring the table
        document.getElementById(`edge_row_${votingTableIdx}`).style.backgroundColor = getColorForEdge(uid_from, uid_to, flag)
    }
    else{
        console.log(`%cERROR! No edge exists between "${uid_from}" and "${uid_to}".`, 'color:red')
    }
}

/*
 * Row Adding Popup
 */

// Manually add a row to the pattern
function addSelectedRowToPattern() {
    let rowSelectedUUID = getSelectedOptionValue('rowAddingSelector')
    let count = getCountForRow(rowSelectedUUID)

    // DOnt ad a duplicate row
    if(!pattern['UID'].includes(rowSelectedUUID)){
        appendRowToPattern(rowSelectedUUID, RATING_UNRATED, 0, [], [], "", isOptional=false)      // Count is 0 because a manually-added row should not have been previously observed in other explanations with the seed rows
        addRowToRate(rowSelectedUUID, count, RATING_UNRATED, isOptional=false)

        refresh(false, removeUnratedRows=false, withoutSearch=true)
    }
    else{
        // TODO: warn user and dont close yet.
        console.log('%cWarning. Row already in pattern. Nothing to add.', 'color:orange')
    }

    changesMadeSinceLastSave = true

    closeRowAddingWindow()    
}

// Populates the row selection based on the query string
function populateRowSelector(){

    let queryString = document.getElementById('rowAddingQueryBox').value
    let queries = queryString.split(' ')

    let selector = document.getElementById('rowAddingSelector')
    let rankedRowsList = []
    for(row of tableRows){
        let text = row.text

        let numWordsInRow = 0
        for(word of queries){
            if(text.indexOf(word) >= 0) numWordsInRow++
        }

        rankedRowsList.push([row, numWordsInRow])
    }

    rankedRowsList = rankedRowsList.sort(function(a, b){
        return b[1] - a[1]
    })

    let htmlStr = ''
    for(rowPkg of rankedRowsList){
        let row = rowPkg[0]
        let numWordsInRow = rowPkg[1]        
        if(numWordsInRow > 0 || queryString.length === 0) htmlStr += `<option value="${row.uuid}">(${numWordsInRow}) [${row.tableName}] ${row.text}</option>\n`
    }

    selector.innerHTML = htmlStr
}

function openRowAddingWindow(){

    // Populate the row selections
    populateRowSelector()

    document.getElementById('rowAddingDiv').style.display = 'flex'
}

function closeRowAddingWindow(){
    document.getElementById('rowAddingDiv').style.display = 'none'
}

/*
 * Hint-adding Popup (for Central-Switchable rows)
 */ 

// On change event for the pivot words textbox
function pivotWordsCallback(){

    let rowIdx = getIndexOfCurrentlySelectedRow();
    let pivotWordsStr = document.getElementById('pivotWordsBox').value

    let hintWords = pivotWordsStr.split(',')

    // Trim so that users can safely add spaces
    for(i in hintWords){
        let word = hintWords[i]
        hintWords[i] = word.trim()
    }
    
    pattern['hintWords'][rowIdx] = hintWords

    changesMadeSinceLastSave = true
}

// On change event for the pivot words textbox
function rowNotesCallback(){

    let rowIdx = getIndexOfCurrentlySelectedRow();
    let rowNotesStr = document.getElementById('rowNotesBox').value
    
    pattern['rowNotes'][rowIdx] = rowNotesStr

    changesMadeSinceLastSave = true
}

function getIndexOfCurrentlySelectedRow() {
    // Find index of currently selected row    
    let rowIdx = parseInt(currentRow.replace("row_", ""))
    let row = rowRatingsForCurrentPattern[rowIdx].row

    // console.log("rowIdx: " + rowIdx)
    // console.log("Row: ")
    // console.log(row)

    for(let i = 0; i < pattern.length; i++){
        // console.log("i=" + i)
        // console.log(pattern['ROW'][i])
        if (pattern['UID'][i] == row.uuid) {
            // console.log("patternIdx: " + i)
            return i;
        }
    }
        
    // console.log("patternIdx: not found")
    return -1;
}
 
function getCurrentlySelectedRow() {
    // Find index of currently selected row    
    let rowIdx = parseInt(currentRow.replace("row_", ""))
    return rowRatingsForCurrentPattern[rowIdx].row
}
 
function getCurrentlySelectedUID() {
    // Find index of currently selected row    
    let rowIdx = parseInt(currentRow.replace("row_", ""))
    return rowRatingsForCurrentPattern[rowIdx].row.uuid
}
 
function addSelectedRowToHint() {
    // console.log(pattern)

    let rowSelectedUUID = getSelectedOptionValue('rowAddingSelector1')
    let count = getCountForRow(rowSelectedUUID)

    let rowIdx = getIndexOfCurrentlySelectedRow();

    // Dont add a duplicate row    
    if (!pattern["hintRowUUIDs"][rowIdx].includes(rowSelectedUUID)) {
        pattern["hintRowUUIDs"][rowIdx].push(rowSelectedUUID);
        populateHintTable();
    } else {
        // TODO: warn user and dont close yet.
        console.log('%cWarning. Row already in hints. Nothing to add.', 'color:orange')
    }

    changesMadeSinceLastSave = true    
    updateHintRowsOnNode(pattern['UID'][rowIdx])

    // Handle the warning messages
    let rowEle = $(`#row_${rowIdx}`).children('td').last()[0]
    markWarningsOnRow(rowEle, rowSelectedUUID)

    //closeRowAddingWindow()    
}

// Make the given hint row the root row of the given node UID
function setHintAsRootRow(hintUID, rootUID){

    // Verify that the pattern does not already include the row we are trying to set as root
    if( !pattern['UID'].includes(hintUID) ){

        // Step 0 - Preliminary Calculations
    
        // Get tableRow for the hint row
        let hintTableRow = getTableRow(hintUID)
    
        // Get the delimited text for the hint row
        let delText = getTextDelFromTableRow(hintTableRow)
        let cleanText = getTextFromTableRow(hintTableRow)
    
        // Step 1 - Update pattern dataframe
    
        // Update the pattern dataframe
        let foundRootRowInPattern = false
        for(let i = 0; i < pattern.length; i++){
            if(pattern['UID'][i] == rootUID){
                idx = i
                let hintUIDs = pattern['hintRowUUIDs'][i]
    
                // Mark down the length for debugging
                let hintUIDs_length = hintUIDs.length
    
                // Filter out the row to be moved (Check that it was removed)
                hintUIDs = hintUIDs.filter(uid => uid != hintUID)
                if(hintUIDs.length < hintUIDs_length){
                    
                    // Add the root row the the hint list
                    hintUIDs.push(rootUID)
    
                    // Write back to the pattern df with new update
                    pattern['UID'][i] = hintUID
                    pattern['hintRowUUIDs'][i] = hintUIDs
                    pattern['ROW'][i] = delText
    
                    // Break the loop
                    foundRootRowInPattern = true
                    break
                }
                else{
                    console.log(`%c\tERROR! There exists no row "${hintUID}" for slot "${rootUID}" in the pattern.`, 'color:red')
                    console.log(hintUIDs)
                    throw new Error(`ERROR! There exists no row "${hintUID}" for slot "${rootUID}" in the pattern.`)
                }
            }
        }
    
        // Pattern updated, move on to graph
        if(foundRootRowInPattern){
    
            // Step 2 - Update the rowRatings Datastructure
            let foundRootRowInRowRatings = false
            for(let i in rowRatingsForCurrentPattern){
                let {row} = rowRatingsForCurrentPattern[i]
                if(row.uuid == rootUID){
                    row.text = cleanText
                    row.textDel = delText
                    row.uuid = hintUID
    
                    // Break the loop
                    foundRootRowInRowRatings = true
                    break
                }
            }
    
            // Second datastructure updated, move on the the graph
            if(foundRootRowInRowRatings){
            
                // Step 3 - Update cytoscape graph
        
                // Get Node and update it with updater method
                // let node = cy.nodes(node => node.id() == rootUID)
                updateNodeState(rootUID, prevRating='', rating='', optional=false, newUID=hintUID)
        
                // Step 4 - Refresh HTML
    
                // Refresh the window
                closeHintAddingWindow()
                openHintAddingWindow()
        
                // Update the HTML
                populatePatternHTML()
        
                // Populate pattern overlap window
                calculateHintOverlap()
            }
            else{
                console.log(`%c\tERROR! There exists no row ${rootUID}" in the current pattern rowRatings (parallel structure).`, 'color:red')
                console.log(getTableRow(rootUID))
                throw new Error(`ERROR! There exists no row "${rootUID}" in the pattern rowRatings (parallel structure).`)
            }
        }
        else{
            console.log(`%c\tERROR! There exists no row ${rootUID}" in the current pattern.`, 'color:red')
            console.log(getTableRow(rootUID))
            throw new Error(`ERROR! There exists no row "${rootUID}" in the pattern.`)
        }
    }
    else{
        document.getElementById('edgeCleanerWarning').innerHTML = `ERROR! There already exists a row "${hintUID}" in pattern.`
        document.getElementById('edgeCleanerWarning').style.display = 'flex'
        console.log(`%c\tERROR! There exists a row "${hintUID}" in pattern.`, 'color:red')
        console.log(hintUIDs)
    }
}

// 
function removeHintCallback(uuidToRemove) {   
    let rowIdx = getIndexOfCurrentlySelectedRow();

    for (let i=0; i<pattern["hintRowUUIDs"][rowIdx].length; i++) {
        if (pattern["hintRowUUIDs"][rowIdx][i] == uuidToRemove) {
            // Remove UUID from hint list
            pattern["hintRowUUIDs"][rowIdx].splice(i, 1);

            // Refresh hint table
            populateHintTable()

            changesMadeSinceLastSave = true
            updateHintRowsOnNode(pattern['UID'][rowIdx])

            // Handle the warning messages
            let rowEle = $(`#row_${rowIdx}`).children('td').last()[0]
            markWarningsOnRow(rowEle, uuidToRemove)

            return;
        }
    }
}

function populateHintTable() {
    // console.log(pattern);
    let patternIdx = getIndexOfCurrentlySelectedRow();

    let root_uid = pattern["UID"][patternIdx]
    let root_row = getTableRow(root_uid)
    let root_cleanText = getTextFromTableRow(root_row)

    let node = cy.nodes(node => node.id() == root_uid)
    let warningRowIdxs = node.data("warningRowIdxs")
    if(typeof warningRowIdxs == 'undefined') warningRowIdxs = []

    var htmlOutStr = "";
    for (var i=0; i<pattern["hintRowUUIDs"][patternIdx].length; i++) {
        
        let hintUID = pattern["hintRowUUIDs"][patternIdx][i]
        var row = getTableRow(hintUID)
        let cleanText = getTextFromTableRow(row)

        htmlOutStr += `<tr> 
            <td>${i}</td>
            <td> <button class="button bad" onclick="removeHintCallback('${hintUID}')"> <i class="fas fa-minus-square"></i> </button> </td>
            <td> <button class="button functional" onclick="setHintAsRootRow('${hintUID}', '${root_uid}')"> <i class="fas fa-plus-square"></i> </button> </td>
            `
        htmlOutStr += (i == warningRowIdxs)? `<td>(Warning) ${cleanText} [${row.tablename}]</td>` : `<td>${cleanText} [${row.tablename}]</td>`
        htmlOutStr += "</tr>"
    }

    $("#rowTextStr").html(`${root_cleanText} [${root_row.tablename}]`)

    $('#substiteRowTable').html(`
    <thead>
    <tr>
        <th>#</th><th>Remove</th><th>Set as Root</th><th>Row Text</th>
    </tr>
    </thead>
    <tbody>
    ` + htmlOutStr + "<tbody>")
}

// Populates the row selection based on the query string
function populateRowSelectorHint() {

    let rowSelected = getCurrentlySelectedRow();
    let tableName = rowSelected.tableName
    let uidRoot = rowSelected.uuid

    // Store the last table name, if valid -- or retrieve the last table name, if empty.  
    // This is so that this function can be called in the HTML on query word updates without having to know the seed tableName (but is a bit hacky)
    if (tableName.length == 0) {
        tableName = lastTableName
    } else {
        lastTableName = tableName
    }

    let queryString = document.getElementById('rowAddingQueryBox1').value
    let queries = queryString.split(' ')

    let selector = document.getElementById('rowAddingSelector1')
    let rankedRowsList = []

    for(row of tableRows){
        let text = row.text        

        if (row.tableName == tableName) {
            let numWordsInRow = 0
            for (word of queries){
                if(text.indexOf(word) >= 0) numWordsInRow++
            }

            rankedRowsList.push([row, numWordsInRow])
        }
    }

    rankedRowsList = rankedRowsList.sort((a, b) => b[1] - a[1])

    // Populate the string with the found rows
    // NOTE skip the first result, as that will be the identical row
    let htmlStr = ''
    for(let i = 0; i < rankedRowsList.length; i++){
        let rowPkg = rankedRowsList[i]
        let row = rowPkg[0]
        let uid = row.uuid
        let numWordsInRow = rowPkg[1]
        if(uid !== uidRoot){ 
            if(numWordsInRow > 0 || queryString.length === 0) htmlStr += `<option value="${uid}">(${numWordsInRow}) [${row.tableName}] ${row.text}</option>\n`
        }
    }

    selector.innerHTML = htmlStr
} 

function getTextStopwordsRemoved(inStr) {
    let words = inStr.replace(/[^\w\d\s]/g, '').split(" ")
    let out = new Set()
    for (let i=0; i<words.length; i++) {
        if (words[i].trim().length > 0) {
           if (!stopWords.includes( words[i].toLowerCase() )) {
                out.add(words[i])
            }
        }
    }

    return Array.from(out);
}

function openEdgeCleanerWindow(){
    
    // Get row 
    let rowIdx = getIndexOfCurrentlySelectedRow();
    let uid = pattern["UID"][rowIdx]
    let row = getTableRow(uid)
    let cleanText = getTextFromTableRow(row)

    // Hint row adding popup is not needed anymore
    closeHintAddingWindow()

    // Populate row text in popup
    $('#rowTextStr_edgeCleaner').html(`${cleanText} [${row.tablename}]`)

    // handles the edge voting table
    populateEdgeTableHTML()

    // Set the div to be visible
    document.getElementById('edgeCleaningDiv').style.display = 'flex'
}

function closeEdgeCleanerWindow(){
    
    // Set the div to be invisible
    document.getElementById('edgeCleaningDiv').style.display = 'none'
}

function openHintAddingWindow() {
    // Ensure that the variables for hints exist in the datastructure
    if (!pattern.hasOwnProperty('hintRowUUIDs')) {        
        pattern['hintRowUUIDs'] = [];
        for (let a=0; a<pattern.length; a++) { pattern['hintRowUUIDs'].push([]) }
        pattern.header.push('hintRowUUIDs')
    }
    
    if (!pattern.hasOwnProperty('hintWords')) {        
        pattern['hintWords'] = Array(pattern.length).fill([])        
        pattern.header.push('hintWords')
    }

    if (!pattern.hasOwnProperty('rowNotes')) {        
        pattern['rowNotes'] = Array(pattern.length).fill("")        
        pattern.header.push('rowNotes')
    }

    // Get row
    let rowIdx = getIndexOfCurrentlySelectedRow();
    let uid = pattern["UID"][rowIdx]
    let rating = pattern["RATING"][rowIdx]
    let row = getTableRow(uid)
    let cleanText = getTextFromTableRow(row)

    // Populate pivot words
    document.getElementById('pivotWordsBox').value = pattern['hintWords'][rowIdx].join(", ")

    // Populate row notes
    document.getElementById('rowNotesBox').value = pattern['rowNotes'][rowIdx]

    // Populate row text    
    $('#rowTextStr').html(`${cleanText} [${row.tablename}]`)

    // Populate initial query text    
    document.getElementById('rowAddingQueryBox1').value = getTextStopwordsRemoved(cleanText).join(" ")

    // Populate the row selections
    populateRowSelectorHint()

    // Populate hint table
    populateHintTable();

    // Display the warning for edgeCleaner not working
    // And toggle button usability
    if(!isSeedRow(rating)) {
        document.getElementById('edgeCleanerWarning').style.display = 'flex'
        document.getElementById("openEdgeCleanerButton").disabled = true
    }
    else{
        document.getElementById("openEdgeCleanerButton").disabled = false
    }

    // Set the div to be visible
    document.getElementById('hintAddingDiv').style.display = 'flex'
}

function closeHintAddingWindow() {
    
    // Hide the div
    document.getElementById('hintAddingDiv').style.display = 'none'

    // Also hide the warning that may hae popped up during use
    document.getElementById('edgeCleanerWarning').style.display = 'none'
    
    // Update the HTML
    populatePatternHTML()

    // Populate pattern overlap window
    calculateHintOverlap()

    changesMadeSinceLastSave = true
}

/*
 * Hint overlap / Overlap Window
 */

function calculateHintOverlap() {
    var currentPatternUUIDs = getPatternRatedUUIDs(pattern, false);
    var currentPatternUUIDsWithHints = getPatternRatedUUIDs(pattern, true);
    var scores = [] 

    // Calculate overlap
    for (let i=0; i<patterns.length; i++) {
        if(patternIdx != i){   // Dont compare to ourselves
            var queryPatternUUIDsWithHints = getPatternRatedUUIDs(patterns[i], true)
            var queryPatternUUIDs = getPatternRatedUUIDs(patterns[i], true)
    
            var diff1 = difference(currentPatternUUIDs, queryPatternUUIDsWithHints).size / currentPatternUUIDs.size
            var diff2 = difference(queryPatternUUIDs, currentPatternUUIDsWithHints).size / queryPatternUUIDs.size
            
            scores.push([i, diff1, diff2]);
        } 
    }

    var sorted = scores.sort(function(a, b) { return Math.min(a[1] - b[1], a[2] - b[2]) });     //##

    // Populate html
    let htmlStr = `
    <table>
    <thead>
        <tr><th> Rank </th> <th> Jump </th> <th> Diff1 </th> <th> Diff2 </th> <th> Pattern Rows </th></tr>
    </thead>
    <tbody>
    `

    // Show ranked list
    let sortedPatternIdxs = [] // list of patternIndices used in the table (for the button callbacks after compilation)

    for (var i=0; i<Math.min(20, sorted.length); i++) {
        //console.log("Rank " + i + ": " + sorted[i][0] + ", " + sorted[i][1].toFixed(2) + ", " + sorted[i][2].toFixed(2) );
        let localPatternIdx = sorted[i][0]
        var patternName = patternNames[localPatternIdx]

        var overlap1 = sorted[i][1]
        var overlap2 = sorted[i][2]
        
        if ((overlap1 != 1.0) && (overlap2 != 1.0)) {

            // Append to the indecies datastructure
            sortedPatternIdxs.push(localPatternIdx)
            
            // Get pattern Flags
            let localPattern = patterns[localPatternIdx]
            let isDone = localPattern.isDone
            let isGood = localPattern.isGood
            let isUncertain = localPattern.isUncertain
            let isBad = localPattern.isBad
            let isRedundant = localPattern.isRedundant

            // Display pattern UUIDs
            var patternUUIDsToDisplay = Array.from(getPatternRatedUUIDs(patterns[sorted[i][0]], false));        

            htmlStr += `\t<tr>
            <td>${i}</td>
            <td> <button class="button hoverable jumper"> <i class="fas fa-arrow-right"></i> </button> </td>
            <td>${overlap1.toFixed(2)}</td>
            <td>${overlap2.toFixed(2)}</td>        
            `

            htmlStr += '<td class="align-left">'
            
            htmlStr += `<div style=\"float:left;width:80%;\"><b>${patternName}</b></div>`

            htmlStr += `<div style="float:right; width=20%; display:flex; justify-content:space-between; margin: 0px 0px 5px 0px;">`
            htmlStr += (isDone)? '<i style="color: #000000" class="far fa-check-circle"></i>' : '<i style="color: #d1d1d1" class="far fa-check-circle"></i>';
            htmlStr += (isGood)? '<i style="color: #000000" class="far fa-thumbs-up"></i>' : '<i style="color: #d1d1d1" class="far fa-thumbs-up"></i>';
            htmlStr += (isUncertain)? '<i style="color: #000000" class="far fa-question-circle"></i>' : '<i style="color: #d1d1d1" class="far fa-question-circle"></i>';
            htmlStr += (isBad)? '<i style="color: #000000" class="far fa-frown"></i>' : '<i style="color: #d1d1d1" class="far fa-frown"></i>';
            htmlStr += (isRedundant)? '<i style="color: #000000" class="far fa-clone"></i>' : '<i style="color: #d1d1d1" class="far fa-clone"></i>';
            htmlStr += "</div><br>"

            for (var j=0; j<patternUUIDsToDisplay.length; j++) {
                let uid_toDisplay = patternUUIDsToDisplay[j]

                let row = getTableRow(uid_toDisplay)
                let cleanText = getTextFromTableRow(row)

                if (currentPatternUUIDsWithHints.has(uid_toDisplay)) {
                    htmlStr += "<font color=\"#ff9896\"><b>"
                    htmlStr += j + ": " + cleanText + "<br>"
                    htmlStr += "</b></font>"
                } 
                else {
                    htmlStr += j + ": " + cleanText + "<br>"
                }
            }
            htmlStr += "</td>"

            htmlStr += `</tr>\n`
        }
    }


    htmlStr += `
    </tbody>
    </table>
    `

    // Populate
    ovWindow.document.getElementById('titleDiv').innerHTML = `<h2>Pattern Overlap View (${patternNames[patternIdx].split('_').join(' ')})</h2>`
    ovWindow.document.getElementById('tableDiv').innerHTML = htmlStr

    // add listeners to buttons now so that the code is run in this scope
    let buttons = $('button.jumper', ovWindow.document)
    for(let i = 0; i < buttons.length; i++){
        let button = $(buttons[i])
        let localPatternIdx = sortedPatternIdxs[i]
        button.click(function(){
            showDone = true
            showGood = true
            showUncertain = true
            showBad = true
            showRedundant = true
            showUnmarked = true
            styleFilterButtons()
            populatePatternSeletor()
            selectPattern("", localPatternIdx)
        })
    }
}

// Helper function: Returns a list of the UUIDs in a pattern that have a rating of "central", "centralsw", or "grounding"
function getPatternRatedUUIDs(patternIn, includeHints) {
    var out = new Set();
    for (var i=0; i<patternIn['UID'].length; i++) {
        var uuid = patternIn['UID'][i];
        var rating = patternIn['RATING'][i];

        if ((rating == RATING_CENTRAL) || (rating == RATING_CENTRALSW) || (rating == RATING_GROUNDING)) {
            out.add(uuid);
            if (includeHints == true) {
                if (patternIn.hasOwnProperty('hintRowUUIDs')) {                
                    var hintUUIDs = patternIn['hintRowUUIDs'][i]
                    for (var j=0; j<hintUUIDs.length; j++) {
                        out.add(hintUUIDs[j])
                    }
                }
            }
        }
    }

    return out;
}

/*
 * Set functions (from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set)
 */
function intersection(setA, setB) {
    var _intersection = new Set();
    for (var elem of setB) {
        if (setA.has(elem)) {
            _intersection.add(elem);
        }
    }
    return _intersection;
}

function difference(setA, setB) {
    var _difference = new Set(setA);
    for (var elem of setB) {
        _difference.delete(elem);
    }
    return _difference;
}

/*
 * Questions View Window
 */

// Populates the questions relevant page
function populateQuestionsWindow(){

    // let numQs = questions.length
    // console.log(`numQs ${numQs}`)

    // Populate a list of questions whose explanations contain any rows from the seed rows
    let seedRowUUIDs = getSeedRowUIDs(includeLexGlue=true)
    let relevantQuestions = []
    for(let i = 0; i < questions.length; i++){
        let question = questions[i]

        let explanationRows = question.explanationRows
        // let answerKey = question.answerKey
        // let explanation = question.explanation
        // let flags = question.flags
        // let qid = question.qid
        // let text = question.text
        // let topic = question.topic

        let explRowsInSeedRows = []
        for(explRow of explanationRows){
            // let role = explRow.role
            // let row = explRow.row
            // let tableName = explRow.tableName
            // let text = explRow.text
            let uuid = explRow.uuid

            //
            if(seedRowUUIDs.includes(uuid)) explRowsInSeedRows.push(explRow.text)
        }

        // Add non-zero counts to the list
        if(explRowsInSeedRows.length > 0) relevantQuestions.push([question, explRowsInSeedRows])
    }

    // Sort by overlap with pattern
    relevantQuestions = relevantQuestions.sort((a, b) => (b[1].length - a[1].length))

    // Populate html
    let htmlStr = `
    <table>
    <thead>
        <tr><th> # </th> <th> Topic </th> <th> QID </th> <th> Row Overlap </th> <th> Question </th> <th> Answer </th></tr>
    </thead>
    <tbody>
    `
    
    for(let i = 0; i < relevantQuestions.length; i++){
        let q = relevantQuestions[i][0]
        let qText = q.text
        let answerKey = `(${q.answerKey})`

        let endOfQuestion = qText.indexOf('(A)')
        if(endOfQuestion < 0){
            endOfQuestion = qText.indexOf('(1)')
        }
        let qTextClean = qText.substring(0, endOfQuestion)

        var correctAnswerIdx = -1;
        var correctAnswer = q.answerKey.toUpperCase()
        if ((correctAnswer.charCodeAt(0) >= 65) && (correctAnswer.charCodeAt(0) <= 69)) {
            // A-E
            correctAnswerIdx = correctAnswer.charCodeAt(0) - 65;
        } else if ((correctAnswer.charCodeAt(0) >= 49) && (correctAnswer.charCodeAt(0) <= 53)) {
            // 1-5
            correctAnswerIdx = correctAnswer.charCodeAt(0) - 49;
        } else {
            // Unrecognized format for answer key
            console.log("ERROR: Unrecognized format for answer key: " + correctAnswer)
        }

        let answer = qText.substring(endOfQuestion).trim()        
        var answerCandidates = answer.split(/\([A-E]\)|\([1-5]\)/g)      
        answerCandidates = answerCandidates.filter(n => n.length>0)     // NOTE: Unclear why, but the first element of the split is often an extra blank
        var answerCandidateHtmlStr = ""

        for (var a=0; a<answerCandidates.length; a++) {            
            if (a == correctAnswerIdx) {
                answerCandidateHtmlStr += "<font color=\"black\">"
            } else {
                answerCandidateHtmlStr += "<font color=\"lightgrey\">"
            }
            answerCandidateHtmlStr += "(" + String.fromCharCode(65+a) + ") "
            answerCandidateHtmlStr += answerCandidates[a]
            answerCandidateHtmlStr += "</font>  "
        }

        htmlStr += `\t<tr><td>${i}</td><td>${q.topic.split('_').join(' ')}</td><td>${q.qid.split('_').join(' ')}</td>
                    <td class="countCol" data-tooltip="asasd">${relevantQuestions[i][1].length} / ${q.explanation.split(" ").length}</td><td>${qTextClean}</td>
                    <td>${answerCandidateHtmlStr}</td></tr>\n`
    }
    htmlStr += `
    </tbody>
    </table>
    `

    // Populate
    rqWindow.document.getElementById('titleDiv').innerHTML = `<h2>Questions Related to Central and Grounding seed rows of pattern "${patternNames[patternIdx].split('_').join(' ')}"</h2>`
    rqWindow.document.getElementById('tableDiv').innerHTML = htmlStr
    sizeRQTable()   // Initial sizing for the table since it is dynamic
}

// Dynamically update the size of the question examples table to fit the page
function sizeRQTable(){
    let tableBody = $('tbody', rqWindow.document)[0]
    if(tableBody) tableBody.style['max-height'] = $( rqWindow ).height() - 175 + "px"
}

/*
 * Query for new rows using seed rows
 */
// Input: 
// seedRowUUIDs: A list of UUIDs in the seed rows
// questions: A list of all questions loaded from tablestore.loadQuestions()
// Output:
// Returns a hashmap of (UUID, count) pairs that should be included. 
// The count refers to how many question's explanations contained that row.
function querySeedRows(seedRowUUIDs) {
    var resultUUIDs = new Map();
    var questionsIncluded = new Set();

    // For each seed row
    for (var i = 0; i < seedRowUUIDs.length; i++) {
        var seedRowUUID = seedRowUUIDs[i];
        // console.log("SEEDUUID: " + seedRowUUID)

        // For each question
        for (var j = 0; j < questions.length; j++) {
            var question = questions[j];

            // If we haven't already included the rows from this question
            if (!questionsIncluded.has(question.qid)) {
                // Check whether this seed row is in the explanation for this question
                var found = false;
                for (var k = 0; k < question.explanationRows.length; k++) {
                    if (question.explanationRows[k].uuid == seedRowUUID) {
                        found = true;
                        //break;
                    }
                }
                
                if (found) {
                    for (var k = 0; k < question.explanationRows.length; k++) {
                        var uuid = question.explanationRows[k].uuid;                        
                        if (resultUUIDs.has(uuid)) {
                            // Existing UUID -- increment count
                            resultUUIDs.set(uuid, resultUUIDs.get(uuid) + 1);
                        } else {
                            // New UUID
                            resultUUIDs.set(uuid, 1);
                        }
                        
                        //## BUG TEST
                        //var count = getCountForRow(uuid)

                    }
                    questionsIncluded.add(question.qid)
                }
                
            }
        }
    }

    // Return
    return resultUUIDs;
}

/*
 * Core active element
 */

// Repopulate html, and find new rows from user voted rows
function refresh(includeGrounding, removeUnratedRows=false, withoutSearch=false) {
    // Dont run if asked not to
    if(!withoutSearch){

        // Determine the curent seed rows 
        // let includeHintRows = document.getElementById("includeHintsInRefresh").checked;
        let seedRows = getSeedRowUIDs(includeGrounding, true)

        // Find potential pattern rows by querying the question explanations
        let resultUIDs = querySeedRows(seedRows)

        // Remove all previously unrated rows (we'll add them in again shortly)
        // This is incase the user has switched between 'includeCentral' and 'includeCentralAndGrounding' modes -- this will remove all the ones the user hasn't rated
        // (effectively decreasing the noise)
        var onlyRated = []
        let existingRowUUIDs = []


        // Store values that should be persistant in 'pattern' before clearing        
        let prevHintUUIDs = new Map()
        let prevHintWords = new Map()
        let prevHintNotes = new Map()
        let prevOptional = new Map()
        for (let i=0; i<pattern['UID'].length; i++) {
            let uuid = pattern['UID'][i];
            let hintRowUUIDs = pattern['hintRowUUIDs'][i];
            let hintWords = pattern['hintWords'][i];
            let hintNotes = pattern['rowNotes'][i];
            let isOptional = pattern['OPTIONAL'][i];

            prevHintUUIDs.set(uuid, hintRowUUIDs);
            prevHintWords.set(uuid, hintWords);
            prevHintNotes.set(uuid, hintNotes);
            prevOptional.set(uuid, isOptional);
        }
        // Now OK to clear pattern
        clearPattern()

        for (var i=0; i<rowRatingsForCurrentPattern.length; i++) {
            if (rowRatingsForCurrentPattern[i].rating != RATING_UNRATED) {
                let uuid = rowRatingsForCurrentPattern[i].row.uuid
                onlyRated.push(rowRatingsForCurrentPattern[i])
                existingRowUUIDs.push(rowRatingsForCurrentPattern[i].row.uuid)
                //## appendRowToPattern(rowRatingsForCurrentPattern[i].row.uuid, rowRatingsForCurrentPattern[i].rating, -1, pattern["hintRowUUIDs"], pattern["hintWords"])  // -1 as a temporary count -- populated below
                appendRowToPattern(rowRatingsForCurrentPattern[i].row.uuid, rowRatingsForCurrentPattern[i].rating, -1, prevHintUUIDs.get(uuid), prevHintWords.get(uuid), prevHintNotes.get(uuid), prevOptional.get(uuid))  // -1 as a temporary count -- populated below
            }
        }
        rowRatingsForCurrentPattern = onlyRated;

        // Update the pattern
        for (keyValuePair of resultUIDs) {
            // console.log(keyValuePair)
            var uuid = keyValuePair[0]
            var count = keyValuePair[1]
            
            // Append it to the list with an unranked rating
            if (!existingRowUUIDs.includes(uuid)) {
                if (!removeUnratedRows) {
                    addRowToRate(uuid, count, RATING_UNRATED, isOptional=false)
                    appendRowToPattern(uuid, RATING_UNRATED, count, [], [], "")
                }
            } else {
                // If row is not new, we can still update the count
                for (var i=0; i<rowRatingsForCurrentPattern.length; i++) {
                    if (rowRatingsForCurrentPattern[i].row.uuid == uuid) {
                        rowRatingsForCurrentPattern[i].count = count;
                        setCountInPattern(uuid, count)
                        break;
                    }
                }
            }
        }    
    }

    // Update the HTML
    currentRow = "row_0"
    populatePatternHTML()

    changesMadeSinceLastSave = true

    // console.log('changesMadeSinceLastSave = true')
}

/*
 * Callbacks
 */

function patternQueryCallback(){
    
    // read the query string into a temp var
    let temp = document.getElementById('patternQueryBox').value
    
    // Determine if the user started using the query box or just cleared it
    let noLongerEmpty = patternQuery.length === 0 && temp.length > 0
    let isNowEmpty = patternQuery.length > 0 && temp.length === 0
    
    // Now update the global pattern query string
    patternQuery = temp

    // store the last pattern selected so that we can jump back to it after clearing the query box
    if(noLongerEmpty) preQueryPattern = getSelectedOptionValue('patternSelector')

    populatePatternSeletor()
    
    // If we just returned to an empty state, then restore the selection to the prequery selection
    if(isNowEmpty) {
        $('#patternSelector option:selected')[0].selected = false
        $(`#patternSelector option[value="${preQueryPattern}"]`)[0].selected = true
    }

    patternChanged()
}

function prevPattern(){
    closeAll()

    let currOption = $('#patternSelector option:selected')
    let prevOption = currOption[0].previousElementSibling
    if(prevOption){
        let prevPatternIdx = getPatternIdx(prevOption.value)
    
        currOption[0].selected = false
        patternIdx = prevPatternIdx
        prevOption.selected = true
        patternChanged()
    }
}


function getPatternIdx(patternName){
    for(i in patternNames){
        let name = patternNames[i]
        if(patternName === name){
            return i
        }
    }
    return -1
}

// Jump to the next
function nextPattern(){
    // closeAll()
    closeAll_withoutCallbacks()

    let currOption = $('#patternSelector option:selected')
    let nextOption = currOption[0].nextElementSibling
    if(nextOption){
        let nextPatternIdx = getPatternIdx(nextOption.value)
    
        currOption[0].selected = false
        patternIdx = nextPatternIdx
        nextOption.selected = true
        patternChanged()
    }
}

// Return to whichever pattern the user had previously selected
function backPattern() {
    closeAll()

    // (if the stack has one element we cant go back any further)
    if(patternSelectionStack.length > 1){

        // Pop the last pattern off the stack 
        // NOTE we're literally just throwing it away here
        patternSelectionStack.pop()

        // read the pattern name we want to jump to
        // pop it off since it will be added back after calling this function
        let patternName = patternSelectionStack.pop()

        selectPattern(patternName)
    }
}

// Prompt user to change the pattern name, and update server with change
function editPatternName(warningPrompt=""){
    let newName = prompt("Change Name: \n" + warningPrompt, patternNames[patternIdx])

    let nameAlreadyExists = patternNames.includes(newName)

    // Null is a cancel
    if(newName !== null && newName.length > 0 && !nameAlreadyExists){
        // Sanitize name
        newName = newName.replace(/[\/]/g, "-")

        // Tell the server of this change
        save(newName=newName, copy=false, whereFrom="editPatternName")        

        // Update the pattern name
        patternNames[patternIdx] = newName
        setPatternName(newName)
        let currOption = $('#patternSelector option:selected')
        currOption.html(newName)
        currOption.val(newName)
    }
    else{
        if(newName !== null){
            let message = ''
            if(nameAlreadyExists) message = `Warning! Pattern "${newName}" already exists. Please choose another pattern`
            if(newName.length == 0) message = `Warning! Cannot save name: "${newName}"`
            console.log(message)
            alert(message)
        }
    }
}

// Prompt user to change the pattern name, and update server with change
function copyPattern(){
    let newName = prompt("New Pattern Name: ", patternNames[patternIdx] + " (copy)")

    // Null is a cancel
    if(newName !== null && newName.length > 0){

        // Tell the server to copy this pattern
        save(newName=newName, copy=true, whereFrom="copyPattern")

        // Add a copy to the patterns list
        patterns.push(deepCopy(pattern))
        patternNames.push(newName)
        populatePatternSeletor()

        // Change page to the new pattern
        $("#patternSelector option:last").attr('selected', 'selected');
        patternChanged()
    }
}

// Callback for when a radio button is pressed
// rowIdx is the name(index) of the button
function voteCallback(rowIdx, rating){

    // Update rating    
    let prevRating = rowRatingsForCurrentPattern[rowIdx].rating
    rowRatingsForCurrentPattern[rowIdx].rating = rating;    

    // This updates the global pattern storage
    // Note, have to find the row first
    let foundRow = false
    let row = rowRatingsForCurrentPattern[rowIdx].row
    let uid = row.uuid
    let text = row.textDel
    for(let i = 0; i < pattern.length; i++){
        if(pattern['UID'][i] === uid){
            pattern['RATING'][i] = rating
            foundRow = true
        }
    }

    // callback handles what happened to the graphical representation of a node on vote
    updateNodeState(uid, prevRating=prevRating, rating=rating)

    // Pattern Not updated
    if(!foundRow){
        console.log('%cERROR! Pattern not updated!', 'color:red')
    }

    // Update coloring
    var rowRefStr = $(`#row_${rowIdx}`)
    colorRowByRating(rowRefStr, rating, uid)

    // Mark that changes have been made that require saving
    changesMadeSinceLastSave = true

    // Update the seed box
    populateHTMLSeedBox()

    populateQuestionsWindow()
}

// Converts a string into a regexp-escaped string suitable for string.replace()
// From https://stackoverflow.com/questions/3446170/escape-string-for-use-in-javascript-regex
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

// Callback for when a radio button is pressed
// rowIdx is the name(index) of the button
function voteCallbackOptional(rowIdx) {
    // Update rating    
    if (rowRatingsForCurrentPattern[rowIdx].optional == true) {
        rowRatingsForCurrentPattern[rowIdx].optional = false;
    } else {
        rowRatingsForCurrentPattern[rowIdx].optional = true;
    }    

    // This updates the global pattern storage
    // Note, have to find the row first
    let row = rowRatingsForCurrentPattern[rowIdx].row
    let uid = row.uuid
    let text = row.textDel
    for(let i = 0; i < pattern.length; i++){
        if(pattern['UID'][i] === uid){
            pattern['OPTIONAL'][i] = rowRatingsForCurrentPattern[rowIdx].optional
        }
    }

    // Change the row text to show it as either 'optional' or non-optional. 
    // Here we update the <TD> row text directly, rather than regenerating the entire table, so that the view doesn't shift. 
    var str = document.getElementById(`row_${rowIdx}_text`).innerHTML;    
    // console.log("Str (before): " + str)
    if (rowRatingsForCurrentPattern[rowIdx].optional == true) {
        console.log("true")
        str = str.replace(escapeRegExp(OPTIONAL_ROW_STR), "")        // Remove any previous optional markings, just in case        
        str += OPTIONAL_ROW_STR                     // Add optional string text to the end of the row text
    } else {
        console.log("false")
        str = str.replace(escapeRegExp(OPTIONAL_ROW_STR), "")       // Remove any previous instance of the row being marked optional in the text description        
    }    
    // console.log("Str (after): " + str)
    document.getElementById(`row_${rowIdx}_text`).innerHTML = str;


    // Mark that changes have been made that require saving
    changesMadeSinceLastSave = true

    // Update the seed box
    populateHTMLSeedBox()

    populateQuestionsWindow()

    // callback handles what happened to the graphical representation of a node on vote
    updateNodeState(uid, optional=rowRatingsForCurrentPattern[rowIdx].optional)
}

// Callback for when the dataset selector is changed
function datasetChanged() {
    
    currentDataset = getSelectedOptionValue("datasetSelector")
    
    patternSelectionStack = []

    var query = {
        selectedDataset: currentDataset,
    }

    // Get the list of patterns available in the dataset
    $.get("/GetPatterns", query, function(data, status){
        if(status === 'success'){
            patterns = data.patterns
            patternNames = data.patternNames
            edgeTablesMap = data.edgeTablesMap

            console.log("PATTERNS: ")
            console.log(patterns)
            populatePatternSeletor()

            //** select first pattern
            patternChanged()
        } 
        else{
            console.log("\tERROR! Failed to retrieve pattern selections from server.")
        }
    })
}

// Callback for whenever a pattern selection changes
function patternChanged(){    
    
    // Close any open poppups
    closeAll_withoutCallbacks()

    currentPattern = getSelectedOptionValue("patternSelector")

    // Function handles updating the idx and setting html 
    selectPattern(currentPattern)
}

/*
 * Statistics
 */
function updateStatistics() {
    
    let numNonZeroRatings = 0;
    for (let i=0; i<rowRatingsForCurrentPattern.length; i++) {
        if (rowRatingsForCurrentPattern[i].rating != RATING_UNRATED) {
            numNonZeroRatings += 1
        }
    }

    let numDonePatterns = 0;
    let numGoodPatterns = 0;
    let numUnsurePatterns = 0;
    let numBadPatterns = 0;
    let numRedundantPatterns = 0;
    let numUnMarkedPatterns = 0;
    for(i in patterns){
        let pat = patterns[i]
        if(pat.isDone) numDonePatterns++
        if(pat.isGood) numGoodPatterns++
        if(pat.isUncertain) numUnsurePatterns++
        if(pat.isBad) numBadPatterns++
        if(pat.isRedundant) numRedundantPatterns++
        if(!pat.isDone && !pat.isGood && !pat.isUncertain && !pat.isBad && !pat.isRedundant) numUnMarkedPatterns++
    }

    $('#tableCounter').html(tableNames.size)
    $('#tableRowCounter').html(tableRows.length)
    $('#questionsCounter').html(questions.length)
    $('#votingRowsCounter').html(rowRatingsForCurrentPattern.length)
    $('#votingRowsLabeledCounter').html(numNonZeroRatings)
    $('#numDonePatterns').html(numDonePatterns)
    $('#numGoodPatterns').html(numGoodPatterns)
    $('#numUnsurePatterns').html(numUnsurePatterns)
    $('#numBadPatterns').html(numBadPatterns)
    $('#numUnMarkedPatterns').html(numUnMarkedPatterns)
    $('#totalPatternCount').html(patterns.length - numRedundantPatterns)
}

// Closes all open popups
function closeAll(){
    closeHintAddingWindow()
    closeRowAddingWindow()
    closeEdgeCleanerWindow()
    closeIMLEditingWindow()
}

function closeAll_withoutCallbacks(){
    
    // closeHintAddingWindow()
    document.getElementById('hintAddingDiv').style.display = 'none'
    document.getElementById('edgeCleanerWarning').style.display = 'none'

    // closeRowAddingWindow()
    document.getElementById('rowAddingDiv').style.display = 'none'

    // closeEdgeCleanerWindow()
    document.getElementById('edgeCleaningDiv').style.display = 'none'
    
    // closeIMLEditingWindow()
    document.getElementById("imlVarView").style.display = 'none'
    document.getElementById("imlCodeBlock").style.display = 'none'

    inATextBox = false
    dataInEditor = false
}

// creates a lookuptable hashmap for the tablestore
function populateTableMap(){
    for(let pkg of tableRows){
        let uid = pkg.uuid
        let textDel = pkg.textDel
        tableMap[uid] = textDel
    }
}

// creates a lookuptable hashmap for the annotatedd tablestore
function populateAnnotatedTableMap(){
    for(let pkg of tableRowsAnnotated){
        let uid = pkg.uuid
        tableMapAnnotated[uid] = pkg
    }
}

/*
 * Annotated tablestore
 */ 



function getTextDelByUID(uid){
    return getTextDelFromTableRow(getTableRow(uid))
}

// Returns the delimited text from the tableRow
function getTextDelFromTableRow(tableRow){
    return tableRow.cells.join(" | ")
}

function getTextByUID(uid){
    return getTextFromTableRow(getTableRow(uid))
}

// Returns the clean text from the tableRow
function getTextFromTableRow(tableRow){
    let {header, cells} = tableRow
    let strOut = ""
    for(let i in tableRow.cells) {
        let cell = cells[i]

        // Determine if cell has content
        if(cell.length > 0){
            let head = header[i]

            // Determine if cell is to be skipped
            if(!head.startsWith('[SKIP]')){
                strOut += ` ${cell}`
            }
        }
    }

    // Cut off leading space and return
    return strOut.substring(1)
}

// Returns a verbose string detailing the columns given sections of text are in
function getLabeledTextFromTableRow(tableRow){
    let {header, headerSanitized, cellLemmas, cellTags} = tableRow
    
    let strOut = ""
    for(let i in cellLemmas) {
        let head = header[i]

        if(isDataCol(head)){
            let lemmas_alts = cellLemmas[i]
            let tags_alts = cellTags[i]
            let head_clean = headerSanitized[i]

            let colStr = ""
            for(let j in lemmas_alts){
                let lemmas = lemmas_alts[j]
                let tags = tags_alts[j]

                for(let k in lemmas){
                    let tag = tags[k]

                    if(isContentTag(tag)){
                        let lemma = lemmas[k]
                        colStr += `${lemma} `
                    }
                }

                // Remove trailing space and add a ; to seperate alts
                colStr = colStr.substring(0, colStr.length-1) + ";"
            }

            // Remove trailing ;
            colStr = colStr.substring(0, colStr.length-1)

            // If the cell has content add it to the hintText
            if(colStr.length > 0) strOut += `${head_clean}:"${colStr}", `
        }
    }

    // Remove trailing comma space
    if(strOut.length > 0) strOut = strOut.substring(0, strOut.length-2)
    
    return strOut
}

function getTableRow(uid){
    if(tableMapAnnotated.hasOwnProperty(uid)){
        return tableMapAnnotated[uid]
    }
    else{
        console.log(`%cERROR! No such row "${uid}" in tablestore.`, "color:red")
        console.log(tableMap[uid])
        throw Error(`ERROR! No such row "${uid}" in tablestore.`)
    }
}

// Set of lemmas in the row
function getLemmaSetFromRow(tableRow, includeNonContentTags=false){
    let lemmaSet = new Set()
    let cellLemmas = tableRow.cellLemmas
    let cellTags = tableRow.cellTags
    for(let i in cellLemmas){
        let cell = cellLemmas[i]
        for(let j in cell){
            let alt = cell[j]
            for(let k in alt){
                let tag = cellTags[i][j][k]
                if(includeNonContentTags || isContentTag(tag)){
                    let lemma = alt[k]
                    lemmaSet.add(lemma)
                }

            }
        }
    }

    return lemmaSet
}

// 
function loadTablestore(cb) {
    // console.log("* Loading TableStore from server...")    

    // Change 'refresh tablestore' button to disabled, to show the tablestore is being refreshed
    document.getElementById("refreshTablestoreButton").classList.remove("unclicked");
    document.getElementById("refreshTablestoreButton").classList.add("disabled");

    // Send GET request to server for tablestore, questions, and related information
    $.get("/GetTablestore", function(data_tablestore, status_tablestore) {        
        tableRows = data_tablestore.rows
        tableRowsAnnotated = data_tablestore.rowsAnnotated
        questions = data_tablestore.questions
        rowRoleFreqs = data_tablestore.rowRoleFreqs
        lemmatizer = data_tablestore.lemmatizer
        tableHeaders = data_tablestore.tableHeaders
        console.log(data_tablestore)

        console.log("First annotated tablestore row:")
        if (typeof tableRowsAnnotated !== 'undefined') {
            console.log(tableRowsAnnotated[0]);
        } else {
            console.log("Annotation not received -- annotating the tablestore is likely disabled in the server.")
        }

        // Populates a quick lookup map of the tablestore
        populateTableMap()
        populateAnnotatedTableMap()

        // Store a global Set of the table names, in case it's ever needed
        // console.log(tableRows)
        for (var i=0; i<tableRows.length; i++) {
            tableNames.add( tableRows[i].tableName );
        }

        // Change 'refresh tablestore' button to normal state, to signify the tablestore has been loaded
        document.getElementById("refreshTablestoreButton").classList.remove("disabled");
        document.getElementById("refreshTablestoreButton").classList.add("unclicked");                

        // If a callback was specified, call it. 
        if (typeof cb !== 'undefined') {
            cb();
        }
    });

}

// Callback for refresh button
function refreshTablestore() {        
    // Load Tablestore
    loadTablestore(function() {
        // Update Statistics
        updateStatistics();

    });

}

// Send a signal to the server requesting an export of the current dataset
function exportData(){
    $.post('/Export', {}, function(res, status){
        console.log(res)
    })
}

/*
 * IML (running scripts)
 */

 // Send a signal to the server requesting an export of the current dataset
 function runPattern() {
    console.log("runPattern(): Started... ")

    // Step 1: Ask the server to save the pattern to IML
    save(name="", copy=false, whereFrom="runPattern")    

    // Step 2: Ask the server to run the IML pattern
    let selectedDataset = getSelectedOptionValue("datasetSelector")
    if(selectedDataset != "ERR") {

        let patternName = patternNames[patternIdx]
    
        // Parse runtime limit box for runtime limit for IML interpreter
        let timeLimit = 0;
        let timeLimitInt = parseInt(document.getElementById('runTimeLimit').value);
        if (!isNaN(timeLimitInt)) timeLimit = timeLimitInt    
    
        // Open the debug info page
        setTimeout(function() {
            //var tableMatchesFilename = "runscript/output/infpatdebugexport-" + patternName.replace(/[^A-Za-z0-9]/g, "") + ".html"
            var tableMatchesFilename = "runscript/output/infpatdebugexport.html"
            var imlOutputWindow1 = window.open(tableMatchesFilename)
            //imlOutputWindow1.focus();        
        }, 2000);
    
        // console.log(`Sending approx. : ${parseInt(dataStr.length / 1000)} KBytes of data`)
        let data = {
            patternName: patternName,
            dataset: selectedDataset,
            timeLimit: timeLimit,
        }
        $.post("/RunPattern", data, function(res, status){
            console.log(res)
            changesMadeSinceLastSave = false
    
            // Open the HTML that displays the eneumerations 
            var imlOutputWindow = window.open("runscript/output/runscript.infpat.html")
            imlOutputWindow.focus();
        })    
    }
    else{
        console.log("%c\tERROR! Failed to read dataset from dataset selector.", "color:red")
        throw Error("ERROR! Failed to read dataset from dataset selector.")
    }
}



/*
 * IML
 */

// Update the IML statstructure on change
function dep_changeToIMLExpression(classHash){
    let expression = document.getElementById(`exprInput_${classHash}`).value
    userIMLEdits[classHash].expression = expression
    changesSinceLastIMLPopulate = true
    return
}

// Flag when a given list of lemmas contains a stopword from the special stopwords list
function anyLemmaInStopWords(lemmas){
    for(let lemma of lemmas){
        if(stopWords_special.includes(lemma)) return true
    }
    return false
}

// Flag when a given list of lemmas contains a stopword from the special stopwords list
function allLemmaInStopWords(lemmas){
    for(let lemma of lemmas){
        if(!stopWords_special.includes(lemma)) return false
    }
    return true
}

function createVarName(lemma, varNum){
    return `<${lemma.replace(/[^A-Za-z]/g, "")}_${varNum}>`
}

// Pulls out the number form the auto generated variable names, or returns 0 if the variable name is not the standard
function getVarNum(varName){
    let str = varName.substring(1, varName.length-1)
    let [_, varNum] = str.split("_")
    if(isNumber(varNum)) return varNum
    else return -1
}

// Clean table header name
function sanitizeColName(name) {
    var str = name.replace(/[^A-Za-z0-9 ]/g, " ")   // Remove all non-alphanumeric, non-space characters -- replace them with spaces
    str = str.replace(/ +/g, " ").trim()            // Truncate multiple spaces to a single space
    str = str.replace(/ /g, "_")                    // Replace spaces with underscores

    return str
}

// Overly simple for now, 
// NOTE: may need more logic is sorting is an issue
function strigifyLemmas(lemmas){
    return lemmas.join(" ")
}

// Deterine from the expression what class this constraint is
function getClassFromExpression(expression){
    console.log(`getClassFromExpression(${expression})`)
    let isLex = expression.includes('\"')
    let isVar = expression.includes('<')
    let isStop = false // TODO
    
    if(isStop)              return CLASS_STOPWORD
    else if(isLex && isVar) return CLASS_VARIABLIZED_LEXICALIZED
    else if(isLex)          return CLASS_LEXICALIZED
    else if(isVar)          return CLASS_VARIABLIZED
    else                    return CLASS_UNRATED
}

function mkUserEditsObj(classes, expression, edgeText){
    return {
        classes: classes,
        default_classes: classes,
        edgeText: edgeText,
        expression: expression,
        default_expression: expression,
    }
}

// adds class tags and other IML related info to the cy edge objects
function classifyGraphConstraints(){

    // Map of edge class info from previous work
    let userIMLEdits = {}

    // Generate lookup maps for quick info
    let ratingMap = createRatingsMap()
    let hintRowsMap = createHintRowsMap()

    // Go through each constraint of each edges and classify them
    // If they are already marked with a class just read off the userIMLEdits map
    // Oherwise determine classification and add to the hash map()
    let varNameMap = {} // Track varnames for a given variablized lemma
    let varNum = 0
    let edges = cy.edges()
    for(let i = 0; i < edges.length; i++){
        let edge = edges[i]
        let edgeID = edge.id()
        let uid_src = edge.data('source')
        let uid_tgt = edge.data('target')
        let constraints = edge.data('constraints')

        let hintRows_src = hintRowsMap[uid_src]
        // let hintRows_tgt = hintRowsMap[uid_tgt]

        // let textCols_src = tableMap[uid_src].split(" | ")
        // let textCols_tgt = tableMap[uid_tgt].split(" | ")

        let src_isSwappable = isSwappableRow(ratingMap[uid_src])
        let tgt_isSwappable = isSwappableRow(ratingMap[uid_tgt])

        for(let j in constraints){
            let {colIdx_from, colIdx_to, on, lemmas, classes, classHash} = constraints[j]
            if(on){
                if(classHash == 0){

                    // Determine classes and var name 
                    let classes = 0
                    let expression = ""
                    let edgeText = ""

                    // space the lemas in a string
                    let lemmaStr = lemmas.join(" ")
    
                    // Both swappable, the constraint is likely variablized
                    if(src_isSwappable && tgt_isSwappable){
        
                        // Initialize the set of constant lemmas to be the lemmas on the edge
                        let constantLemmaSet = new Set(lemmas)
                        let altLemmasList = [lemmas]       // List of all lemmas used across the hint rows
        
                        // go through the hint rows and determine whether 
                        // there exists constant lemma accross all rows with intersections
                        for(let {uid} of hintRows_src){
                            let text = tableMap[uid]
                            let cols = text.split(' | ')
                            let col_from = cols[colIdx_from]
                            let lemmas_from = tokenize(col_from, withLemmatize=true)
        
                            constantLemmaSet = intersection(constantLemmaSet, new Set(lemmas_from))
                            altLemmasList.push(lemmas_from)
                        }
    
                        // Filter the constant lemmas out of each lemmasset
                        for(let k in altLemmasList){
                            let altLemmas = altLemmasList[k]
                            altLemmas = altLemmas.filter(a => !constantLemmaSet.has(a))
                            altLemmasList[k] = altLemmas
                        }
    
                        altLemmasList = removeDuplicateRows(altLemmasList)
        
                        // Store these lemma sets into the object
                        constraints[j].altLemmasList = altLemmasList
                        constraints[j].constantLemmaSet = constantLemmaSet
        
                        // All lemmas are constant
                        let numConstantLemmas = constantLemmaSet.size
                        let numLemmas = (new Set(lemmas)).size
                        if(numConstantLemmas == numLemmas){
                            classes = CLASS_LEXICALIZED
                            expression = `"${lemmaStr}"`
                            edgeText = lemmaStr
                        }
                        // No constants => variable
                        else {
                            if(numConstantLemmas == 0)  classes = CLASS_VARIABLIZED
                            else                        classes = CLASS_VARIABLIZED_LEXICALIZED
        
                            // Create an ordered compund class string (e.g. <var1> + "sith" or "wet" + <var2>)
                            let expressions = []
                            for(lemma of lemmas){
                                // If lemma is in the constant set then just keep it as string
                                if(constantLemmaSet.has(lemma)) expressions.push(`"${lemma}"`)
                                else{
                                    // For the lemma that are variable determine the variable name and push onto stack
                                    let varName = createVarName(lemma, varNum)
                                    if(!varNameMap.hasOwnProperty(lemma)){
                                        varNameMap[lemma] = varName
                                        varNum++
                                    }
                                    else{
                                        varName = varNameMap[lemma]
                                    }
                                    expressions.push(varName)
                                }
                            }
                            
                            // join and merge adjacents
                            expression = expressions.join("+")                // Join expressions with +
                            expression = expression.replace(/\"\+\"/g, " ")   // Merge adjacent strings (e.g. "..."+"..." => "... ...")
                            expression = expression.replace(/\>\+\</g, "_")   // Merge adjacent vars    (e.g. <...>+<...> => <..._...>)
    
                            // Include all alts in the edge string
                            // TODO, make this smarter
                            for(let altLemmmas of altLemmasList){
                                edgeText += `\n${altLemmmas.join(" ")}`
                            }
                            edgeText = edgeText.substring(1)
                        }
                    }
                    else{ // Otherwise constraint will be lexicalized
                        classes = CLASS_LEXICALIZED
                        expression = `"${lemmaStr}"`
                        edgeText = lemmaStr
                    }
    
                    // Store class and var name and generate class hash
                    constraints[j]['classes'] = classes
                    constraints[j]['expression'] = expression
                    let newClassHash = `${expression}`.hashCode()
                    constraints[j]['classHash'] = newClassHash
    
                    // Initialize the edge editing storage object in the pattern
                    let edgeEditingObj = mkUserEditsObj(classes, expression, edgeText)
    
                    // If lemmas contains a stopword then mark the constraints as stopped,
                    // NOTE it should keep the default info for later if the user demarks the stopword
                    if(allLemmaInStopWords(lemmas)){
                        edgeEditingObj.classes = CLASS_STOPWORD
                        edgeEditingObj.expression = `"${lemmaStr}"`
                    }
        
                    userIMLEdits[newClassHash] = edgeEditingObj
                }
                // Hash leads to info in the pattern; use that
                else if(userIMLEdits.hasOwnProperty(classHash)){
                    constraints[j]['classes'] = userIMLEdits[classHash].classes
                    constraints[j]['expression'] = userIMLEdits[classHash].expression
                }
                else console.log(`%cERROR! Edge: "${edgeID}"(${j}) has class hash, "${classHash}" that is not defined in pattern lookup.`, "color:red")
            }
        }
    }

    // Write back to main mem
    userIMLEdits.lemmasToVarNameMap = varNameMap
    pattern.userIMLEdits = userIMLEdits
}

// Get the part of speach tag for a given row, col, lemma
function getPOSTag(uid, colIdx, colLemma){
    for(let tableRow of tableRowsAnnotated){
        if(tableRow.uuid == uid){
            let cellTags = tableRow.cellTags[colIdx]
            let cellLemmas = tableRow.cellLemmas[colIdx]

            // Find the lemma we are interested in, and return its tag
            for(let i in cellLemmas){
                let alt = cellLemmas[i]

                for(let j in alt){
                    let cellLemma = alt[j]
                    if(cellLemma == colLemma) return cellTags[i][j]
                }
            }
        }
    }
    return "UNK"
}

// Determine form the given constraints and the col text 
// what minimum constraint is required to meet all constraints given
function getSchemaCellConstraintFromConstraintList(constraints, row, colIdx, lemmasToVarNameMap_local=null){
    // Parse the row
    let uid = row.uuid
    let cells = row.cellLemmas
    let cellTags = row.cellTags
    let header = row.header

    // This will be returned once populated
    let expression = ""

    // Skip non data-cols
    let head = header[colIdx]
    if(isDataCol(head)){

        // Get row rating
        let rating = getRoleForUIDInPattern(uid)
        let isSwappable = isSwappableRow(rating)

        // get the info maps, or use the ones already passed in
        let lemmasToVarNameMap = (lemmasToVarNameMap_local == null)? pattern.userIMLEdits.lemmasToVarNameMap : lemmasToVarNameMap_local

        // Determine the constraint state of each lemma in each constraint
        let lemmaConstraintMap = {}
        let lemmaPOSMap = {}
        for(let {lemmas, classes, tags} of constraints){
            for(let i in lemmas){
                let lemma = lemmas[i]
                if(!lemmaConstraintMap.hasOwnProperty(lemma)) {
                    if(classes == CLASS_LEXICALIZED) lemmaConstraintMap[lemma] = CLASS_LEXICALIZED
                    if(classes == CLASS_POS){
                        lemmaConstraintMap[lemma] = CLASS_POS
                        lemmaPOSMap[lemma] = tags[i]
                        continue    // Dont mark as var after marking as POS
                    }
                }	

                // Determine if this lemma is variablized 	
                // If it is not in the expression or has a < before it then it is variable	
                if(typeof expression != 'undefined') {
                    let idxOfLemma = expression.indexOf(lemma)	
                    if(idxOfLemma <= 0){	
                        lemmaConstraintMap[lemma] = CLASS_VARIABLIZED	
                    }	
                    else if(expression.substring(idxOfLemma-1, idxOfLemma) != '"') {	
                        lemmaConstraintMap[lemma] = CLASS_VARIABLIZED	
                    }	
                }
            }
        }

        // Loop through each lemma of each alt in order,
        // So that the expressions are determined in order
        // List of expressions for this col
        let alts = cells[colIdx]
        let expressionsByAlt = Array(alts.length)
        let altsUsed = Array(alts.length)   // Keep track of what alts get used, we cant have constraints on both alts
        for(let j = 0; j < alts.length; j++){
            let lemmas = alts[j]
            expressionsByAlt[j] = []
            altsUsed[j] = {numVar:0, numLex:0, numPOSTag: 0, numOptional:0}

            for(let k = 0; k < lemmas.length; k++){
                let lemma = lemmas[k]
                if(isContentTag(cellTags[colIdx][j][k])){

                    // Determine the constraint class for this lemma
                    let lemmaConstraintClass = lemmaConstraintMap[lemma]
                    let lemmaIsConstrained = typeof lemmaConstraintClass != 'undefined'
                    let lemmaIsLexicallyConstrained = lemmaConstraintClass == CLASS_LEXICALIZED
                    let lemmaIsPOSConstrained = lemmaConstraintClass == CLASS_POS
                        
                    // If constrained determine by what means it was constrained
                    if(lemmaIsConstrained){
                        if(lemmaIsLexicallyConstrained){
                            expressionsByAlt[j].push(`"${lemma}"`)
                            altsUsed[j].numLex++
                        }
                        // else if(lemmaIsPOSConstrained){
                        //     // POS constraining disabled
                        //     // expressionsByAlt[j].push(`'POS:${lemmaPOSMap[lemma]}'`)
                        //     // altsUsed[j].numPOSTag++
                        // }
                        else{
                            let varName = lemmasToVarNameMap[lemma]

                            if(varName) {
                                expressionsByAlt[j].push(varName)
                                altsUsed[j].numVar++
                            }
                            else{
                                // console.log(`%cWarning! Trying to variablize lemma "${lemma}" which has not been mapped to a variable name.`, 'color:orange')
                                expressionsByAlt[j].push(`"${lemma}"`)
                                altsUsed[j].numLex++
                                // ** Assume over constraining
                            }
                        }
                    }
                    else{
                        // No constraint
                        // If it is a swappable row mark as optional otherwise lexicalize it
                        if(isSwappable){
                            let posTag = cellTags[colIdx][j][k]
                            expressionsByAlt[j].push(`*'POS:${posTag}'`)    // NOTE: we're using ' so that these dont merge with lexical strings
                            altsUsed[j].numOptional++
                        }else{
                            expressionsByAlt[j].push(`"${lemma}"`)
                            altsUsed[j].numLex++
                        }
                    }

                }
            }
        }

        // console.log('expressionsByAlt')
        // console.log(expressionsByAlt)

        // Trim the alts so that only one is used
        // Alt decision criteria:
        // NumVars > numLex > numOptional
        let altIdxToUse = 0
        let maxAltScore = 0
        for(let j = 0; j < alts.length; j++){
            let {numVar, numLex, numOptional} = altsUsed[j]
            let altScore = 100*numVar + 10*numLex + numOptional
            if(altScore > maxAltScore) {
                altIdxToUse = j
                maxAltScore = altScore
            }
        }
    
        let expressions = expressionsByAlt[altIdxToUse]

        // Prepare and clean the final expression
        expression = expressions.join("+")            // Join expressions with +
        expression = expression.replace(/\"\+\"/g, " ")   // Merge adjacent strings (e.g. "..."+"..." => "... ...")
        expression = expression.replace(/'/g, '"')          // Convert optional ' quotes to "
        // expression = expression.replace(/\>\+\</g, "_")   // Merge adjacent vars    (e.g. <...>+<...> => <..._...>)
    }

    return expression
}

function rankRoleForSort(role){
    switch(role){
        case RATING_CENTRALSW:
            return 6
        case RATING_GROUNDING:
            return 5
        case RATING_CENTRAL:
            return 4
        case RATING_LEXGLUE:
            return 3
        case RATING_MAYBE:
            return 2
        case RATING_UNRATED:
            return 1
        case RATING_BAD:
            return 0
    }
    throw Error(`Invalid role "${role}" passed into rankRoleForSort()`)
}

// Generate IML string for the current pattern
function createIMLFromPattern(){

    // -------------------
    // - Pre-computation -
    // -------------------

    let hintRowMap = createHintRowsMap(includeRootRow=false)
    
    // Read necessary variables into local scope
    let patternName = patternNames[patternIdx]
    let description = pattern.notes.replace(/\"/g, "'")

    // Use the live copy if it is non-empty
    let userIMLEdits = pattern.userIMLEdits

    // console.log('userIMLEdits')
    // console.log(userIMLEdits)

    // Read useful maps
    let lemmasToVarNameMap = userIMLEdits.lemmasToVarNameMap
    // console.log(lemmasToVarNameMap)

    // Clean the pattern name
    let cleanPatternName = sanitizeColName(patternName)

    // -------------------
    // - String Building -
    // -------------------

    // Initialize IML string
    let strOut =    "// Automatically converted pattern\n"
    strOut +=       `inferencepattern ${cleanPatternName}\n\n`

    strOut +=           "\t// Plain text description\n"
    strOut +=           `\tdescription = "${description}"\n\n`

    strOut +=           "\t// Requirements\n\n"

    strOut +=           "\t// Row Definitions\n\n"
    
    // Loop through the nodes for this patterns
    // And generate an IML line for each
    let nodes = cy.nodes()
    nodes = nodes.sort((node_A, node_B) => rankRoleForSort(node_B.data('role')) - rankRoleForSort(node_A.data('role'))) // Sort the node so that the rows are sorted
    for (let n = 0; n < nodes.length; n++) {
        let node = nodes[n]
        let uid_n = node.id()

        // Important variables from the pattern
        let rowIdx = getRowIdx(uid_n)
        let rating = pattern['RATING'][rowIdx]
        let isOptional = pattern['OPTIONAL'][rowIdx]
        let tableName = pattern['TABLE'][rowIdx]
        let hintWords = pattern['hintWords'][rowIdx]
        let notes = pattern['rowNotes'][rowIdx]

        let row_n = getTableRow(uid_n)
        let hintRows_n = hintRowMap[uid_n]
        let tableHeaderSanitized = row_n.headerSanitized
        let tableHeader = row_n.header
        let cells_n = row_n.cellLemmas
        let cellTags_n = row_n.cellTags        

        // Human readable row text
        let cleanText = getTextFromTableRow(row_n)
        let cleanTextVerbose = getLabeledTextFromTableRow(row_n)

        // Comment text
        let optionalText = (isOptional)? "(OPTIONAL)" : ""
        strOut += `\t// ${rating.padEnd(14, ' ')} ${optionalText} ${tableName}: ${cleanText}     (${row_n.uuid}) [ ${cleanTextVerbose} ]\n`
        for (let hintRow of hintRows_n) {
            let hintRowText = getTextFromTableRow(hintRow.tableRow)            
            let hintRowTextVerbose = getLabeledTextFromTableRow(hintRow.tableRow)
            strOut += `\t//        HINTROW ${hintRowText}     (${hintRow.tableRow.uuid}) [ ${hintRowTextVerbose} ]\n`
        }
        if (hintWords.length > 0) {
            let hintWordsStr = hintWords.join(", ")
            strOut += `\t//        Hint Words: ${hintWordsStr}\n`
        }
        if (notes.length > 0) {
            strOut += `\t//  Notes: ${notes}\n`
        }

        // Find all visible edges for which this node is connected
        let edges = cy.edges(function(edge){return (edge.data('source') == uid_n || edge.data('target') == uid_n) && edge.visible()})

        /**
         * Step 2 - 1
         */
        // Loop through the connected edges and append the sort the constraints by colidx in the map "constrainedCols"
        let constrainedCols = {}
        for(let edgeIdx = 0; edgeIdx < edges.length; edgeIdx++){
            let edge = edges[edgeIdx]
            let uid_tgt = edge.data('target')
            let constraints = edge.data('constraints_over')
            let flipEdge = (uid_n == uid_tgt)
            
            for(let constraint of constraints){
                let {colIdx_from, colIdx_to, on} = constraint
                
                if(on){
                    let key = (flipEdge)? colIdx_to : colIdx_from
                    if(constrainedCols.hasOwnProperty(key)) constrainedCols[key].push(constraint)
                    else                                    constrainedCols[key] = [constraint]
                }
            }
        }

        /**
         * Step 2 - 2
         */
        // Determine Slot-level constraints by looking for constant lemmas accross hintrows
        // 3d-loop through the root row and determine what cells share lemmas accross all hintrows

        // For each data cell
        let constantLemmasByCell = Array(cells_n.length)
        let constantPOSByCell = Array(cells_n.length)
        for(let i = 0; i < cells_n.length; i++){
            let head_n = tableHeader[i]
            constantLemmasByCell[i] = {lemmas:[], expression:"", classes: CLASS_LEXICALIZED}
            constantPOSByCell[i] = {lemmas: [], expression:"", classes: CLASS_POS, tags:[]}
            if(isDataCol(head_n)){

                // Determine which alts contain constant lemmas
                // Seperate by alt so that we only pick one alt in this
                let alts_n = cells_n[i]
                let constantLemmasByAlt = Array(alts_n.length)
                let constantPOSByAlt = Array(alts_n.length)
                for(let j = 0; j < alts_n.length; j++){
                    let lemmas_n = alts_n[j]

                    // For each lemma in the alt 
                    // Loop through the hint rows and look for this lemma in all rows
                    let constantLemmasByAlt_j = []
                    let constantPOSTagsByAlt_j = []
                    let constantPOSRootLemmaaByAlt_j = []
                    for(let k = 0; k < lemmas_n.length; k++){
                        let lemma_n = lemmas_n[k]
                        let tag_n = cellTags_n[i][j][k]
                        if(isContentTag(tag_n)){
    
                            let lemmaInAllRows = true
                            let posTagInAllRows = true
                            for(let hintRow of hintRows_n){
                                let tableRow_n = hintRow.tableRow
                                let hintCellLemmas_n = tableRow_n.cellLemmas
                                let hintCellPOSTags_n = tableRow_n.cellTags

                                // NOTE: Not being able to read this address in the hint cells is also a failure
                                // I'm using try for clarity: i could also do if(i < len && j < a[i].len...) that sounds gross
                                
                                // Lemma
                                try{
                                    let hintLemma = hintCellLemmas_n[i][j][k]

                                    if(hintLemma != lemma_n){
                                        lemmaInAllRows = false
                                        break
                                    }
                                }
                                catch(e){
                                    lemmaInAllRows = false
                                }

                                // POS tag
                                try{
                                    let hintPOSTag = hintCellPOSTags_n[i][j][k]

                                    if(hintPOSTag != tag_n){
                                        posTagInAllRows = false
                                        break
                                    }
                                }
                                catch(e){
                                    posTagInAllRows = false
                                }
                            }

                            if(lemmaInAllRows) constantLemmasByAlt_j.push(lemma_n)
                            if(posTagInAllRows) {
                                constantPOSTagsByAlt_j.push(tag_n)
                                constantPOSRootLemmaaByAlt_j.push(lemma_n)
                            }
                        }
                    }

                    constantLemmasByAlt[j] = {lemmas:constantLemmasByAlt_j, expression:`"${constantLemmasByAlt_j.join(" ")}"`, classes: CLASS_LEXICALIZED}
                    constantPOSByAlt[j] = {lemmas: constantPOSRootLemmaaByAlt_j, expression:`"${constantPOSTagsByAlt_j.join(" ")}"`, classes: CLASS_POS, tags:constantPOSTagsByAlt_j}
                }

                // Narrow the constant lemma for this cell down to the alt with the most constant lemmas
                let mostConstLemmasAltIdx = 0
                let mostConstLemmasAltCnt = 0
                for(let j in constantLemmasByAlt){
                    let {lemmas} = constantLemmasByAlt[j]
                    let constantLemmasCnt = lemmas.length
                    if(constantLemmasCnt > mostConstLemmasAltCnt){
                        mostConstLemmasAltCnt = constantLemmasCnt
                        mostConstLemmasAltIdx = j
                    }
                }

                let mostConstPOSAltIdx = 0
                let mostConstPOSAltCnt = 0
                for(let j in constantPOSByAlt){
                    let {tags} = constantPOSByAlt[j]
                    let constantPOSCnt = tags.length
                    if(constantPOSCnt > mostConstPOSAltCnt){
                        mostConstPOSAltCnt = constantPOSCnt
                        mostConstPOSAltIdx = j
                    }
                }

                // Save that list of constant lemmas to the cell level array
                constantLemmasByCell[i] = constantLemmasByAlt[mostConstLemmasAltIdx]
                constantPOSByCell[i] = constantPOSByAlt[mostConstPOSAltIdx]
            }
        }

        // Go through each cell, 
        // If there exists a non-zero number of constant lemmas in that cell &&
        // there are no constranints already put on that cell, then 

        // Lemmas
        for(let i in constantLemmasByCell){
            let constraint = constantLemmasByCell[i]
            let i_str = ""+i    // Might need this... idk
            if(constraint.lemmas.length > 0){
                if(!constrainedCols.hasOwnProperty(i_str)){
                    constrainedCols[i_str] = [constraint]
                }
                else{
                    constrainedCols[i_str].push(constraint)
                }
            }
        }

        // POS Tags
        // for(let i in constantPOSByCell){
        //     let constraint = constantPOSByCell[i]
        //     let i_str = ""+i    // Might need this... idk
        //     if(constraint.tags.length > 0){
        //         if(!constrainedCols.hasOwnProperty(i_str)){
        //             constrainedCols[i_str] = [constraint]
        //         }
        //         else{
        //             constrainedCols[i_str].push(constraint)
        //         }
        //     }
        // }

        // Sort the constrained cols for user
        let constrainedColIndicies = getObjKeys(constrainedCols)
        constrainedColIndicies = constrainedColIndicies.sort((a,b) => a - b)

        // If there are no constraints for this node, show a warning and comment out the following line
        if (constrainedColIndicies.length <= 0) {
            console.log('%cWarning! Node with no visible edges.', 'color:orange')
            strOut += "//WARNING! Slot with no constraints.\n//"
        }

        // Add UUID (for static rows) or table (for dynamic rows)
        strOut += (isOptional)? `\t//row *autoRow${n} = [` : `\t//row autoRow${n} = [`
        strOut += (isSwappableRow(rating))? `TABLE:"${tableName}"` : `UUID:"${uid_n}"`
        for (let colIdx of constrainedColIndicies){
            let constraints = constrainedCols[colIdx]
            let colName = tableHeaderSanitized[colIdx]

            // From the list of constraints on this column determine the required constraint
            let expression = getSchemaCellConstraintFromConstraintList(constraints, row_n, colIdx, lemmasToVarNameMap_local=lemmasToVarNameMap)
            let edgeStr = `${colName}: ${expression}`

            strOut += `, ${edgeStr}`
        }
        strOut += '] //OVER-CONSTR.\n'


        /**
         * Step 1 - 1
         */
        // Loop through the connected edges and append the sort the constraints by colidx in the map "constrainedCols"
        constrainedCols = {}
        for(let edgeIdx = 0; edgeIdx < edges.length; edgeIdx++){
            let edge = edges[edgeIdx]
            let uid_tgt = edge.data('target')
            let constraints = edge.data('constraints')
            let flipEdge = (uid_n == uid_tgt)
            
            for(let constraint of constraints){
                let {colIdx_from, colIdx_to, on} = constraint
                
                if(on){
                    let key = (flipEdge)? colIdx_to : colIdx_from
                    if(constrainedCols.hasOwnProperty(key)) constrainedCols[key].push(constraint)
                    else                                    constrainedCols[key] = [constraint]
                }
            }
        }

        /**
         * Step 1 - 2
         */
        // Determine Slot-level constraints by looking for constant lemmas accross hintrows
        // 3d-loop through the root row and determine what cells share lemmas accross all hintrows

        // For each data cell
        constantLemmasByCell = Array(cells_n.length)
        constantPOSByCell = Array(cells_n.length)
        for(let i = 0; i < cells_n.length; i++){
            let head_n = tableHeader[i]
            constantLemmasByCell[i] = {lemmas:[], expression:"", classes: CLASS_LEXICALIZED}
            constantPOSByCell[i] = {lemmas: [], expression:"", classes: CLASS_POS, tags:[]}
            if(isDataCol(head_n)){

                // Determine which alts contain constant lemmas
                // Seperate by alt so that we only pick one alt in this
                let alts_n = cells_n[i]
                let constantLemmasByAlt = Array(alts_n.length)
                let constantPOSByAlt = Array(alts_n.length)
                for(let j = 0; j < alts_n.length; j++){
                    let lemmas_n = alts_n[j]

                    // For each lemma in the alt 
                    // Loop through the hint rows and look for this lemma in all rows
                    let constantLemmasByAlt_j = []
                    let constantPOSTagsByAlt_j = []
                    let constantPOSRootLemmaaByAlt_j = []
                    for(let k = 0; k < lemmas_n.length; k++){
                        let lemma_n = lemmas_n[k]
                        let tag_n = cellTags_n[i][j][k]
                        if(isContentTag(tag_n)){
    
                            let lemmaInAllRows = true
                            let posTagInAllRows = true
                            for(let hintRow of hintRows_n){
                                let tableRow_n = hintRow.tableRow
                                let hintCellLemmas_n = tableRow_n.cellLemmas
                                let hintCellPOSTags_n = tableRow_n.cellTags

                                // NOTE: Not being able to read this address in the hint cells is also a failure
                                // I'm using try for clarity: i could also do if(i < len && j < a[i].len...) that sounds gross
                                
                                // Lemma
                                try{
                                    let hintLemma = hintCellLemmas_n[i][j][k]

                                    if(hintLemma != lemma_n){
                                        lemmaInAllRows = false
                                        break
                                    }
                                }
                                catch(e){
                                    lemmaInAllRows = false
                                }

                                // POS tag
                                try{
                                    let hintPOSTag = hintCellPOSTags_n[i][j][k]

                                    if(hintPOSTag != tag_n){
                                        posTagInAllRows = false
                                        break
                                    }
                                }
                                catch(e){
                                    posTagInAllRows = false
                                }
                            }

                            if(lemmaInAllRows) constantLemmasByAlt_j.push(lemma_n)
                            if(posTagInAllRows) {
                                constantPOSTagsByAlt_j.push(tag_n)
                                constantPOSRootLemmaaByAlt_j.push(lemma_n)
                            }
                        }
                    }

                    constantLemmasByAlt[j] = {lemmas:constantLemmasByAlt_j, expression:`"${constantLemmasByAlt_j.join(" ")}"`, classes: CLASS_LEXICALIZED}
                    constantPOSByAlt[j] = {lemmas: constantPOSRootLemmaaByAlt_j, expression:`"${constantPOSTagsByAlt_j.join(" ")}"`, classes: CLASS_POS, tags:constantPOSTagsByAlt_j}
                }

                // Narrow the constant lemma for this cell down to the alt with the most constant lemmas
                let mostConstLemmasAltIdx = 0
                let mostConstLemmasAltCnt = 0
                for(let j in constantLemmasByAlt){
                    let {lemmas} = constantLemmasByAlt[j]
                    let constantLemmasCnt = lemmas.length
                    if(constantLemmasCnt > mostConstLemmasAltCnt){
                        mostConstLemmasAltCnt = constantLemmasCnt
                        mostConstLemmasAltIdx = j
                    }
                }

                let mostConstPOSAltIdx = 0
                let mostConstPOSAltCnt = 0
                for(let j in constantPOSByAlt){
                    let {tags} = constantPOSByAlt[j]
                    let constantPOSCnt = tags.length
                    if(constantPOSCnt > mostConstPOSAltCnt){
                        mostConstPOSAltCnt = constantPOSCnt
                        mostConstPOSAltIdx = j
                    }
                }

                // Save that list of constant lemmas to the cell level array
                constantLemmasByCell[i] = constantLemmasByAlt[mostConstLemmasAltIdx]
                constantPOSByCell[i] = constantPOSByAlt[mostConstPOSAltIdx]
            }
        }

        // Go through each cell, 
        // If there exists a non-zero number of constant lemmas in that cell &&
        // there are no constranints already put on that cell, then 

        // Lemmas
        for(let i in constantLemmasByCell){
            let constraint = constantLemmasByCell[i]
            let i_str = ""+i    // Might need this... idk
            if(constraint.lemmas.length > 0){
                if(!constrainedCols.hasOwnProperty(i_str)){
                    constrainedCols[i_str] = [constraint]
                }
                else{
                    constrainedCols[i_str].push(constraint)
                }
            }
        }

        // POS Tags
        // for(let i in constantPOSByCell){
        //     let constraint = constantPOSByCell[i]
        //     let i_str = ""+i    // Might need this... idk
        //     if(constraint.tags.length > 0){
        //         if(!constrainedCols.hasOwnProperty(i_str)){
        //             constrainedCols[i_str] = [constraint]
        //         }
        //         else{
        //             constrainedCols[i_str].push(constraint)
        //         }
        //     }
        // }

        // Sort the constrained cols for user
        constrainedColIndicies = getObjKeys(constrainedCols)
        constrainedColIndicies = constrainedColIndicies.sort((a,b) => a - b)

        // If there are no constraints for this node, show a warning and comment out the following line
        if (constrainedColIndicies.length <= 0) {
            console.log('%cWarning! Node with no visible edges.', 'color:orange')
            strOut += "//WARNING! Slot with no constraints.\n//"
        }

        // Add UUID (for static rows) or table (for dynamic rows)
        strOut += (isOptional)? `\trow *autoRow${n} = [` : `\trow autoRow${n} = [`
        strOut += (isSwappableRow(rating))? `TABLE:"${tableName}"` : `UUID:"${uid_n}"`
        for (let colIdx of constrainedColIndicies){
            let constraints = constrainedCols[colIdx]
            let colName = tableHeaderSanitized[colIdx]

            // From the list of constraints on this column determine the required constraint
            let expression = getSchemaCellConstraintFromConstraintList(constraints, row_n, colIdx, lemmasToVarNameMap_local=lemmasToVarNameMap)
            let edgeStr = `${colName}: ${expression}`

            strOut += `, ${edgeStr}`
        }
        strOut += ']\n\n'

    }
  
    // Step N: Close pattern
    strOut += "endinferencepattern\n"

    // console.log(strOut)

    return strOut
} 

// Helper function to determine give a var num hasnt been used
function getNextVarNum(lemmasToVarNameMap){
    let highestVarNum = -1
    let lemmasToVarNameItter = Object.entries(lemmasToVarNameMap)
    for(let [lemmaStr_i, varName_i] of lemmasToVarNameItter){
        let varNum = getVarNum(varName_i)
        if(varNum > highestVarNum) highestVarNum = varNum
    }
    highestVarNum++
    return highestVarNum
}

// Generate the var name for the given set of lemmas
function getConstraintVarName(lemmas){

    // Stack for the var names we find for each lemma
    let varNames = []

    // Map of var names for known lemmas
    let lemmasToVarNameMap = userIMLEdits.lemmasToVarNameMap

    // For each lemma lookup the given var name
    for (let lemmaStr of lemmas){
        let varName = lemmasToVarNameMap[lemmaStr]
        
        // If a variable name has not been given yet, then deermine a new one by finding the highest var num (e.g. var_X) 
        if(typeof varName == 'undefined'){
            
            let varNum = getNextVarNum(lemmasToVarNameMap)

            varName = createVarName(lemmaStr, varNum)
    
            // save the var name to the map
            lemmasToVarNameMap[lemmaStr] = varName
            userIMLEdits.lemmasToVarNameMap = lemmasToVarNameMap
        }

        varNames.push(varName)
    }

    // Join the list together and cut off the >< 
    let fullVarName = varNames.join("+")

    return fullVarName
}

// Callback for button vote to classify a constraint
function dep_clfConstraint(classHash, classes){

    /**
     * Button Highlight
     */
    let userIMLEdit = userIMLEdits_live[classHash]

    // Read what the prev class was 
    let currClass = userIMLEdit.classes

    // Find and style the button
    $(`#button_${classHash}_${currClass}`).removeClass(`selected`)
    $(`#button_${classHash}_${classes}`).addClass(`selected`)

    /**
     * Safety Check
     */

    // Warning messages
    let lexToVarWarningMsg = "1- Warning. Setting static edge to dynamic."
    let warningMessageDisplayed = document.getElementById("userIMLEditingWarningP").innerHTML
    // If the row was marked as static
    let default_classes = userIMLEdit.default_classes
    if(default_classes == CLASS_LEXICALIZED && // Default class was static
        classes != CLASS_LEXICALIZED &&     // And we aren't changing to a static
        classes != CLASS_STOPWORD){         // 
        
        // If warning message is not already there then display it and exit
        if(warningMessageDisplayed != lexToVarWarningMsg){
            document.getElementById("userIMLEditingWarningP").innerHTML = lexToVarWarningMsg
            return
        }
    }
    else{
        document.getElementById("userIMLEditingWarningP").innerHTML = "&nbsp;"
    }

    /**
     * Update Internals
     */

    // Write the new class
    userIMLEdit.classes = classes

    if(classes == CLASS_LEXICALIZED || classes == CLASS_STOPWORD) userIMLEdit.expression = "\"" + strigifyLemmas(userIMLEdit.lemmas) + "\""
    if(classes == CLASS_VARIABLIZED) userIMLEdit.expression = getConstraintVarName(userIMLEdit.lemmas)
    if(classes == CLASS_VARIABLIZED_LEXICALIZED) userIMLEdit.expression = getConstraintVarName(userIMLEdit.lemmas)  // TODO user needs to be able to select which ones are variablized

    userIMLEdits_live[classHash] = userIMLEdit
    changesMadeSinceLastSave = true
    changesSinceLastIMLPopulate = true

    populateUserIMLEditingTable()
    populateIMLViewer()
}

// Interface for allowing users to apply manual information in the auto IML generation
function openIMLEditingWindow(){

    // Toggle display
    console.log(document.getElementById("imlVarView").style.display)
    if(document.getElementById("imlVarView").style.display == 'flex'){
        closeIMLEditingWindow()
    }
    else{

        inATextBox = true

        // Populate the two windows
        populateUserIMLEditingTable()
        populateIMLViewer()
    
        document.getElementById("imlVarView").style.display = 'flex'
        document.getElementById("imlCodeBlock").style.display = 'flex'

        // Scroll to top
        editorIML.setScrollPosition({scrollTop: 0});
    }
}

function closeIMLEditingWindow(){

    pattern.iml = editorIML.getValue()

    document.getElementById("imlVarView").style.display = 'none'
    document.getElementById("imlCodeBlock").style.display = 'none'

    inATextBox = false
    dataInEditor = false
}

/*
 * Main entry point
 */

// Execute when the HTML is ready
$(document).ready(function(){

    // Initialize the code editing space 
    // (NOTE: requires us to make is visible while initiaizing)
    document.getElementById("imlCodeBlock").style.display = 'flex'
    require(['vs/editor/editor.main'], function () {

        monaco.languages.register({ id: 'iml' });

        // Register a tokens provider for the language
        monaco.languages.setMonarchTokensProvider('iml', {
            tokenizer: {
                root: [
                    [/\trow/, "keyword"],
                    [/^inferencepattern/, "keyword"],
                    [/autoRow[0-9]+ = \[UUID/, "custom-static-row"],
                    [/autoRow[0-9]+ = \[TABLE/, "custom-dynamic-row"],
                    [/\/\/.*/, "comment"],
                    [/\"[^\"]*\"/, "string"],
                    [/<[^>]*>/, "variable"],
                    [/[a-zA-Z0-9_]*:/, "constraint"],
                ],
            }
        });
        
        // Define a new theme that contains only rules that match this language
        monaco.editor.defineTheme('vs-plus', {
            base: 'vs',
            inherit: true,
            rules: [
                { token: 'constraint', fontStyle: 'italic' },
            ]
        });
        
        let codeBlock = document.getElementById("imlCodeBlock")
        editorIML = monaco.editor.create(codeBlock, {
            // value: "// jadhasdkahasdawa\nvar sand = 'string';\n[DAS:ASDA]\nclass DAVE{}",
            // language: 'javascript',
            // theme: 'vs-plus',
            value: "",
            language: 'iml',
            theme: 'vs-plus',
            lineNumbers: "on",
            roundedSelection: false,
            scrollBeyondLastLine: true,
            readOnly: false,
            // wordWrap: 'on',
            wrappingIndent: "same",
            layoutInfo: {
                
            },
        });

        // Add a listener on code changes
        console.log(editorIML)
        editorIML.onDidChangeModelContent((e) => {
            
            let text = editorIML.getValue()

            if(text != ""){
                pattern.iml = text
            }
            
        });

        document.getElementById("imlCodeBlock").style.display = 'none'
        
    })

    // Request the tablestore
    loadTablestore(function() {

        // console.log("TEST -- Loaded " + tableRows.length + " rows." );
        // Get options for the selectors
        $.get("/GetDatasetSelections", function(data_datasets, status_datasets){
            if(status_datasets === 'success'){
                let possibleDatasets = data_datasets.possibleDatasets
                prefs = data_datasets.prefs
                AUTOSAVE = prefs.autosave

                READY_TO_POLL = true

                // Populate the selector options (string)
                let datasetOptions = ""
                for(possibleDataset of possibleDatasets){
                    datasetOptions += `<option value="${possibleDataset}">${possibleDataset}</option>\n`
                }

                // Post it in the HTML
                $('#datasetSelector').html(datasetOptions)
                
                // ---------------------
                // Parse the preferences
                // ---------------------
                // console.log(prefs)

                // Filter parameters
                let prefShowDone =      prefs.showDone
                let prefShowGood =      prefs.showGood
                let prefShowUncertain = prefs.showUncertain
                let prefShowBad =       prefs.showBad
                let prefShowRedundant = prefs.showRedundant
                let prefShowUnmarked =  prefs.showUnmarked
                if(typeof prefShowDone !== 'undefined') showDone =              prefShowDone
                if(typeof prefShowGood !== 'undefined') showGood =              prefShowGood
                if(typeof prefShowUncertain !== 'undefined') showUncertain =    prefShowUncertain
                if(typeof prefShowBad !== 'undefined') showBad =                prefShowBad
                if(typeof prefShowRedundant !== 'undefined') showRedundant =    prefShowRedundant
                if(typeof prefShowUnmarked !== 'undefined') showUnmarked =      prefShowUnmarked
                styleFilterButtons()

                // Default sorting method
                let defaultSort = prefs.defaultSort
                if(typeof defaultSort !== 'undefined') selectByValue('sortSelector', defaultSort)

                // Select default dataset and run callback
                let defaultDataset = prefs.defaultDataset
                if(typeof defaultDataset !== 'undefined') selectByValue('datasetSelector', defaultDataset)
                datasetChanged()

            }
            else{
                console.log("\tERROR! Failed to retrieve dataset selections from server.")
            }            
        })
    })

    // Attatch tooltips to all elements that use them
    attatchToolTips()

    // Style the filter buttons
    //styleFilterButtons()

    addFlaggerOnTextBoxes()

    // --------
    // Macros -
    // --------
    document.addEventListener("keydown", function(event) {
        
        let keyPressed = event.key
        // console.log(keyPressed)
        let idx = currentRow.substr(4) // Get row idx

        // Things that only trigger if we aren't in a text box
        if(!inATextBox){
            switch(keyPressed){
            // Row Voting (1-5)
                case "1": // Central
                    event.preventDefault()
                    voteCallback(idx, RATING_CENTRAL)   // Vote
                    incrementCurrentRow()               // increment to next row
                    break
                case "2": // Central (Switchable)
                    event.preventDefault()
                    voteCallback(idx, RATING_CENTRALSW) // Vote
                    incrementCurrentRow()               // increment to next row
                    break
                case "3": // Grounding
                    event.preventDefault()
                    voteCallback(idx, RATING_GROUNDING)     // Vote
                    incrementCurrentRow()               // increment to next row
                    break
                case "4": // Lexical Glue
                    event.preventDefault()
                    voteCallback(idx, RATING_LEXGLUE)     // Vote
                    incrementCurrentRow()               // increment to next row
                    break
                case "5": // Maybe
                    event.preventDefault()
                    voteCallback(idx, RATING_MAYBE)     // Vote
                    incrementCurrentRow()               // increment to next row
                    break
                case "6": // Optional
                    event.preventDefault()
                    voteCallbackOptional(idx)   // Vote
                    incrementCurrentRow()               // increment to next row
                    break
                case "7": // Bad
                    event.preventDefault()
                    voteCallback(idx, RATING_BAD)       // Vote
                    incrementCurrentRow()               // increment to next row
                    break
                case "8": // Remove Row Rating
                    event.preventDefault()
                    voteCallback(idx, RATING_UNRATED)   // Vote
                    incrementCurrentRow()               // increment to next row
                    break
            // Mobility (Arrow Keys)
                case "ArrowUp": // Move up a row
                    event.preventDefault()
                    decrementCurrentRow()
                    break
                case "ArrowDown": // Move down a row
                    event.preventDefault()
                    incrementCurrentRow()
                    break
                case "ArrowLeft": // Goto prev pattern
                    event.preventDefault()
                    prevPattern()
                    break
                case "ArrowRight": 
                    event.preventDefault()
                    nextPattern()
                    break
                // case "x": // Rename
                //     if(event.ctrlKey){
                //         event.preventDefault()
                //         editPatternName()
                //     }
                //     break
                // case "c": // Refresh (Central)
                //     if(event.ctrlKey){
                //         copyPattern()
                //     }
                //     break
                default: // Print out keys (For determining key codes)
                    // event.preventDefault()
                    // console.log(event.key)
                    break
            }
        }

        // Things that trigger regardless of textbox status
        switch(keyPressed){
        // Mark pattern (F1-F5)
            case "F1": // DONE
                event.preventDefault()
                markPattern('DONE')
                break
            case "F2": // GOOD
                event.preventDefault()
                markPattern('GOOD')
                break
            case "F3": // UNCERTAIN
                event.preventDefault()
                markPattern('UNCERTAIN')
                break
            case "F4": // BAD
                event.preventDefault()
                markPattern('BAD')
                break
            case "F5": // REDUNDANT
                event.preventDefault()
                markPattern('REDUNDANT')
                break
        // Button HotKeys
            case "r": // Refresh (Central)
                if(event.ctrlKey){
                    event.preventDefault()
                    refresh(false, removeUnratedRows=false)
                }
                break
            case "R": // Refresh (Central + Grounding)
                if(event.ctrlKey){
                    event.preventDefault()
                    refresh(true, removeUnratedRows=false)
                }
                break
            case "a": // Add Row
                if(event.ctrlKey){
                    event.preventDefault()
                    openRowAddingWindow()
                }
                break
            case "s": // Save
                if(event.ctrlKey){
                    event.preventDefault()
                    console.log("manual save")
                    save(name="", copy=false, whereFrom="manual save")                    
                }
                break     
            case "q": // Save
                if(event.ctrlKey){
                    event.preventDefault()
                    
                    console.log("editorIML")
                    console.log(editorIML.wordWrap)
                }
                break                
            case "Escape":
                closeAll()
                break
            default: // Print out keys (For determining key codes)
                // event.preventDefault()
                // console.log(event.key)
                break
        }
    })

    rqWindow = window.open("relevantQuestionView.html", '_blank', 'height=850px,width=1400px, left=10, top=10')
    ovWindow = window.open("overlapView.html", '_blank', 'height=850px,width=1400px, left=60, top=110')
    cyWindow = window.open("cyView.html", '_blank', 'height=850px,width=1400px, left=110, top=210')
    
    // Automatically close the pop-ups on either master page refresh or master page close
    window.onbeforeunload = function(){
        rqWindow.close()
        ovWindow.close()
        cyWindow.close()
    }

    // Ensure that table is always height of window
    $(rqWindow).resize(sizeRQTable)    
    $(window).resize(sizeVotingTable)    

    cyWindow.onload = function() {
    // Tie the cytoscape graph to the 'cy' element in the popup
        cy = cytoscape({
            container: cyWindow.document.getElementById('cy'), // container to render in

            style: cyStyle,

            // interaction options:
            minZoom: 2e-1,
            maxZoom: 1e1,
            zoom: 1,
            "pan": {
                "x": 500,
                "y": 500
            },
            "zoomingEnabled": true,
            "userZoomingEnabled": true,
            "panningEnabled": true,
            "userPanningEnabled": true,
            "boxSelectionEnabled": false,
            "renderer": {
                "name": "canvas"
            }
        })

    }

    // Begin polling logic
    beginPolling()
})
