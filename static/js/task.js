/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

/********************
 * Task Parameters
 *
 * Sets task parameters like number of lists, list length, presentation rate,
 *
 ********************/

// user determined task params
var num_of_lists = 3;
var list_length = 5;
var recall_time = 3000; // number of milleseconds given to recall
var word_pool = make_pool(); // function in utils.js


// fixed task params
var mycondition = condition;  // these two variables are passed by the psiturk server process
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to


// preallocate and initialize variables
var cur_list_num = 0; //counter to keep track of current list --- uses zero indexing


/********************
* HTML manipulation
*
* All HTML files in the templates directory are requested 
* from the server when the PsiTurk object is created above. We
* need code to get those pages from the PsiTurk object and 
* insert them into the document.
*
********************/

// List Instruction html pages
var instructionPages = [
    "instructions/instructions-FR-greeting.html",
    "instructions/instructions-judgment.html",
    "instructions/instructions-FR-final.html",
];

// List of Task html pages
var pages = [
    "stage.html",
    "postquestionnaire.html",
];

// load all pages we will use
psiTurk.preloadPages(instructionPages);
psiTurk.preloadPages(pages);


/********************
 * Free recall task       *
 ********************/
var RunFR = function() {


    /******
     * Setup variables for this list
     *
     ****/
    var cur_phase,
        wordon, // time word is presented
        listening = false,
        first_recall = true, // keep tack of whether this is the first recall for a list
        stims = word_pool.splice(0,list_length); // get the items for this list: the next list_length elements of word pool


    /******
     * Function to advance to the next item or recall depending on the phase of the list
     *
     ****/
    var next = function() {


        if (stims.length===0) { // if there are no stims left, we have entered the recall phase
            if (first_recall) {
                cur_phase = "RECALL"
                start_time = new Date().getTime();
                first_recall = false
            }
            listening = true;
            recall_period()
        }
        else { // otherwise we still have items to present
            cur_phase = "STUDY"
            stim = stims.shift();
            d3.select("#task").html('<p>Would it fit in a shoebox?</p>');
            present_item( stim[0] );
            wordon = new Date().getTime();
            listening = true;
            d3.select("#query").html('<p id="prompt">Type "R" for bigger, "B" for smaller.</p>');
        }
    };


    /******
     * Function to record responses as appropriate for the phase (study vs recall)
     *
     * This control the flow of the experiment. It keeps track of how many words have been presented and starts the
     * recall period when all the words are done
     *
     * Then it keeps track of the number of milliseconds the recall period has been going on and ends when time is up
     *
     * Then it checks how many lists have been done and runs another if there are more left.
     *
     * This is all handled through recursion rather than for loops
     *
     ****/
    var response_handler = function(e) {
        if (!listening) return;

        var keyCode = e.keyCode,
            recalled_item,
            response;

        // handler for the study phase
        if (cur_phase=="STUDY") {

            switch (keyCode) {
                case 66:
                    // "B"
                    response = "bigger";
                    break;
                case 83:
                    // "G"
                    response = "smaller";
                    break;
                default:
                    response = "";
                    break;
            }
            if (response.length > 0) {
                listening = false;
                var rt = new Date().getTime() - wordon;

                psiTurk.recordTrialData({
                        'list': cur_list_num,
                        'phase': "study",
                        'word': stim[0],
                        'response': response,
                        'rt': rt
                    }
                );
                remove_word();
                next();
            }
        }

        // handler for the recall phase
        if (cur_phase === "RECALL") {

            switch (keyCode) {
                case 13:
                    recalled_item = document.getElementById("recall_field").value
                    break;
                // default:
                //     recalled_iem = "9999";
                //     break;
            }
            if (recalled_item.length > 0) {
                listening = false;

                var rt = new Date().getTime() - wordon;
                psiTurk.recordTrialData({
                        'list': cur_list_num,
                        'phase': "recall",
                        'response': recalled_item,
                        'rt': rt
                    }
                );

                var elapsed = new Date().getTime() - start_time;
                var last_list = cur_list_num+1==num_of_lists; // check if we already have presented all the lists

                if (elapsed < recall_time) {
                    next()
                }
                else {

                    if (last_list) {
                        finish()
                    }
                    else {
                        cur_list_num++
                        RunFR()
                    }
                }

            }

        }

    };


    /******
     * Function to end the FR task once all the lists are presented
     *
     ****/
    var finish = function() {
        $("body").unbind("keydown", response_handler); // Unbind keys
        currentview = new Questionnaire();
    };


    /******
     * Function to present an item
     *
     ****/
    var present_item = function(text) {
        d3.select("#stim")
            .append("div")
            .attr("id","word")
            .style("color","black")
            .style("text-align","center")
            .style("font-size","50px")
            .style("font-weight","400")
            .style("margin","20px")
            .text(text);
    };


    /******
     * Function to remove an item from the screen
     *
     ****/
    var remove_word = function() {
        d3.select("#word").remove();
    };


    /******
     * Function to present text box for recalling words
     *
     ****/
    var recall_period = function() {
        //remove text from encoding task
        d3.select("#query").remove();
        d3.select("#task").remove();

        // display input box
        d3.select("#recall_input").html('<span>Type a word and press ENTER to submit:</span> ' +
            '<input type="text" id="recall_field" name="recall_field"/>');

    };



    /******
     * start the task
     *
     ****/
    // Load the stage.html snippet into the body of the page
    psiTurk.showPage('stage.html');

    // Register the response handler that is defined above to handle any
    // key down events.
    $("body").focus().keydown(response_handler);

    // Start the test
    next();
};




/****************
* Questionnaire *
****************/

var Questionnaire = function() {

	var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

	record_responses = function() {

		psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'submit'});

		$('textarea').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);
		});
		$('select').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);		
		});

	};

	prompt_resubmit = function() {
		document.body.innerHTML = error_message;
		$("#resubmit").click(resubmit);
	};

	resubmit = function() {
		document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
		reprompt = setTimeout(prompt_resubmit, 10000);
		
		psiTurk.saveData({
			success: function() {
			    clearInterval(reprompt); 
                psiTurk.computeBonus('compute_bonus', function(){finish()}); 
			}, 
			error: prompt_resubmit
		});
	};

	// Load the questionnaire snippet 
	psiTurk.showPage('postquestionnaire.html');
	psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'begin'});
	
	$("#next").click(function () {
	    record_responses();
	    psiTurk.saveData({
            success: function(){
                psiTurk.computeBonus('compute_bonus', function() { 
                	psiTurk.completeHIT(); // when finished saving compute bonus, the quit
                }); 
            }, 
            error: prompt_resubmit});
	});
    
	
};

// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
$(window).load( function(){
    psiTurk.doInstructions(
    	instructionPages, // a list of pages you want to display in sequence
    	function() { currentview = new RunFR(); } // what you want to do when you are done with instructions
    );
});
