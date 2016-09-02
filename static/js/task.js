/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */



// todo: for implicit task an extra question saying before you started the first list, did you suspect your memory would be tested.
// todo: it is easy to accidently type a number in the first recall box. Make it ignore responses unless box is focus?
// todo: consider doing psiturk.saveData() after each list---with no arguments it hangs
// todo: review ad---do we really want them to see a full version of the consent in the ad?


// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

/********************
 * Task Parameters
 *
 * Sets task parameters like number of lists, list length, presentation rate,
 *
 ********************/

// user determined task params
var num_of_lists = 2;
var list_length = 5;
var pres_rate = 1500; // number of mileseconds each word presented for
var isi = 100; // number of ms of blank screen between word presentations
var recall_time = 10000; // number of milleseconds given to recall
var delay_between_lists = 5000; // number of mileseconds to pause between lists (display get ready message)
var end_distractor_delay = 15000; // number of mileseconds of distraction task before recall
var word_pool = make_pool(); // function in utils.js


// fixed task params
var mycondition = condition;  // these two variables are passed by the psiturk server process
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to
var one_to_nine = [1, 2, 3, 4, 5, 6, 7, 8, 9]; // digits to use in constructing math distractor problems


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
        end_distractor_done = false, // has the end of list distractor been finished yet
        stims = word_pool.splice(0,list_length); // get the items for this list: the next list_length elements of word pool


    /******
     * Function to advance to the next item or recall depending on the phase of the list
     *
     ****/
    var next = function() {

        //figure out which phase of the task we are in

        // what to do if end of list distractor phase
        if (stims.length===0 && !end_distractor_done) { // if there are no stims left, we have entered the recall phase
            if (cur_phase != "DISTRACTOR"){
                remove_word()
                cur_phase = "DISTRACTOR";
                setTimeout(function(){wrapup_end_distractor(); }, end_distractor_delay); // start a timer that will end the distraction period
            }
            end_distractor_task()
        }

        // what to do if recall phase
        else if (stims.length===0 && end_distractor_done) { // if there are no stims left, we have entered the recall phase
            if (first_recall) {
                remove_word()
                cur_phase = "RECALL"
                start_time = new Date().getTime();
            }
            recall_period()
        }

        // what to do if study phase
        else { // otherwise we still have items to present
            cur_phase = "STUDY"
            stim = stims.shift();
            if (stims.length===list_length-1) {
                if (cur_list_num==0) {
                    ready_message = "Get ready! The list will begin shortly. Position your fingers over the 'y' and 'n' keys so you are ready to respond!"
                }
                else {
                    ready_message = "Get ready! A new list will begin shortly. Position your fingers over the 'y' and 'n' keys so you are ready to respond!"
                }
                d3.select("#stim")
                    .append("div")
                    .attr("id","word")
                    .style("color","black")
                    .style("text-align","center")
                    .style("font-size","50px")
                    .style("font-weight","400")
                    .style("margin","20px")
                    .text(ready_message);
                setTimeout(function(){present_item( stim[0] ); }, delay_between_lists);
            }
            else{
                remove_word()
                present_item( stim[0] );
                }

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
            subjects_answer,
            recalled_item,
            response;


        // handler for the study phase
        if (cur_phase === "STUDY") {

            switch (keyCode) {
                case 89:
                    // "Y"
                    response = "smaller";
                    break;
                case 78:
                    // "N"
                    response = "bigger";
                    break;
                default:
                    response = "";
                    d3.select("#query").html('<p id="prompt"> <span style="color: red; ">Invalid Response. Press Y or N</span></p>');
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

                var elapsed = new Date().getTime() - start_time;
                psiTurk.recordTrialData({
                        'list': cur_list_num,
                        'phase': "recall",
                        'response': recalled_item,
                        'rt': elapsed
                    }
                );
                next()

            }

        }


        // handler for the distractor task
        if (cur_phase === "DISTRACTOR") {
            switch (keyCode) {
                case 13:
                    subjects_answer = document.getElementById("recall_field").value
                    break;
                // default:
                //     recalled_iem = "9999";
                //     break;
            }
            if (subjects_answer.length > 0) {
                listening = false;

                var elapsed = new Date().getTime() - end_distractor_start_time;
                psiTurk.recordTrialData({
                        'list': cur_list_num,
                        'phase': "end_distractor",
                        'correct_answer': correct_answer,
                        'subjects_answer': subjects_answer,
                        'rt': elapsed
                    }
                );
                next()

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

        // remove any words already on screen
        remove_word()

        // show the word for pres_rate ms
        d3.select("#task").html('<p>Would it fit in a shoebox?</p>');
        d3.select("#query").html('<p id="prompt">press "Y" for yes, "N" for no.</p>');
        d3.select("#stim")
            .append("div")
            .attr("id","word")
            .style("color","black")
            .style("text-align","center")
            .style("font-size","50px")
            .style("font-weight","400")
            .style("margin","20px")
            .text(text);

        // start listening and record start time
        listening = true;
        wordon = new Date().getTime();

        setTimeout(function(){wrapup_word(); }, pres_rate);

    };

        /******
     * Function to present text box for recalling words
     *
     ****/
    var recall_period = function() {
        //remove text from encoding task
        d3.select("#query").remove();
//        d3.select("#task").remove();

        // display input box
        disp_this = '<p>You now have ' + recall_time/1000 +
            ' seconds to recall as many words as you can, in any order. If you cannot remember any more words, that is okay; the task will automatically advance when the time is up.</p>'
        d3.select("#task").html(disp_this);
        d3.select("#recall_input").html('<span>Type a word and press ENTER to submit:</span> ' +
            '<input type="text" id="recall_field" name="recall_field"/>');

        if (!first_recall) {
            d3.select("#recall_field").node().focus()
        }

        // start listening
        listening = true;

        if (first_recall) {

            setTimeout(function(){wrapup_recall(); }, recall_time);
            first_recall = false
        }

    };

    var end_distractor_task = function() {
        //remove text from encoding task
//        d3.select("#query").remove();
//        d3.select("#task").remove();

        // setup math problem
        shuffled_digits = _.shuffle(one_to_nine);
        A = shuffled_digits[0];
        B = shuffled_digits[1];
        C = shuffled_digits[2];
        correct_answer = A + B + C;
        subjects_answer = [];

        // display task text
        disp_this = '<p>You will now solve math problems for ' + end_distractor_delay/1000 +
            ' seconds. Try to solve as many problems as you can without sacrificing accuracy. The task will automatically advance when the time is up.</p>'
        d3.select("#task").html(disp_this);
        disp_this = '<p><span style="color: black; font-size: 50px">' + A + '+' + B + '+' + C + '=?</span></p>'
        d3.select("#query").html(disp_this);
        d3.select("#recall_input").html('<span>Add the three numbers, type your answer, and press ENTER to submit:</span> ' +
            '<input type="text" id="recall_field" name="recall_field"/>');

        // display input box
        d3.select("#recall_field").node().focus()
        cur_phase = "DISTRACTOR";
        listening = true;
        end_distractor_start_time = new Date().getTime();

    }

    var wrapup_end_distractor = function() {
        end_distractor_done = true;
        next()
    }



    /******
     * Function to record a time out response if no response was made to the word and move on
     *
     ****/
     var wrapup_word = function() {

        // record a time out if they have not responded to the word
        if (listening) {
            var rt = new Date().getTime() - wordon;
            psiTurk.recordTrialData({
                    'list': cur_list_num,
                    'phase': "study",
                    'word': stim[0],
                    'response': "Timed_Out",
                    'rt': rt
                }
            );
        }

        // show a fixation for isi ms
        // remove text from encoding task
        // stop listening
        listening = false;
        d3.select("#word").remove();
        d3.select("#task").html('<p> </p>');
        d3.select("#query").html('<p id="prompt"> </p>');
        d3.select("#stim")
            .append("div")
            .attr("id","word")
            .style("color","black")
            .style("text-align","center")
            .style("font-size","50px")
            .style("font-weight","400")
            .style("margin","20px")
            .text("+");
        setTimeout(function(){next(); }, isi);

     }


         /******
     * Function to end recall period
     *
     ****/
     var wrapup_recall = function() {

        listening = false;
        var last_list = cur_list_num+1==num_of_lists; // check if we already have presented all the lists
        if (last_list) {
            finish()
        }
        else {
           cur_list_num++
           cur_phase = "STUDY"
//           psiturk.saveData()
            RunFR()
        }
     }



























    /******
     * Function to remove an item from the screen
     *
     ****/
    var remove_word = function() {
        d3.select("#word").remove();
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

        // I think someting like this:
		$('input').each( function(i, val) {
		    psiTurk.recordTrialData({
                        'phase': "postquestionnaire",
                        'strategy': this.name,
                        'strategy_description': this.value,
                        'used': this.checked,
                        });

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
