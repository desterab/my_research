/*
 * Requires:
 incorporateurk.js
 *     utils.js
 */


// before going live:
// todo: what about intrusions from referents?
// todo: change to production database
// todo: change version number in config.txt
// todo: review config.txt
// todo: note methods change to one list with longer pres rate and shorter isi to give eaiser time with differing judgment
// todo: note in methods change to size task --- is it bigger rather than would it fit


// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

/********************
 * Task Parameters
 *
 * Sets task parameters like number of lists, list length, presentation rate,
 *
 ********************/

// user determined task params
var num_of_lists = 1;
var list_length = 16;
var word_pres_rate = 1000; // number of milliseconds each word presented for
var distractor_pres_rate = 10000; // number of milliseconds each distractor between words is presented for
var isi = 500; // number of ms of blank screen between word presentations
var recall_time = 75000; // number of milleseconds given to recall
var delay_between_lists = 5000; // number of mileseconds to pause between lists (display get ready message)
var distractor_delay = 16000; // number of mileseconds of distraction task before recall
var recall_box_lag = 1000; // number of ms to ignore input into the text box after recall period starts --- so people don't accidentally enter responses to the math task here
var word_pool = make_pool(); // function in utils.js

// set of objects to compare against
var size_referents = [
    ['<center><p>Is it easy to judge if it is larger than a <strong>golf ball</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>tennis ball</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>bowling ball</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>postage stamp</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>key</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>penny?'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>dollar bill</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>credit card</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>2-liter bottle</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>gallon of milk</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>lightbulb</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>cell phone</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>coffee mug</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>house plant</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>computer mouse</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>text book</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>chair</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>t-shirt</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>taco</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>sandwich</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>beer bottle</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>piano</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>potato</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>watermelon</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>yardstick</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>magazine</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>mosquito</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>thimble</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>tuba</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>violin</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>AA battery</strong>?</p></center>'],
    ['<center><p>Is it easy to judge if it is larger than a <strong>keyboard</strong>?</p></center>'],
];
size_referents = _.shuffle(size_referents);



// fixed task params

//condition is passed by psiturk based on num_conds variable in config.txt runs from 0 to num_conds-1. for this experiment, 0 = explicit, 1 = implicit
var instruction_condition = condition;
//var instruction_condition = 1; // temp divert everyone into implicit cond


// counterbalance is passed by psiturk based on num_counters variable in config.txt runs from 0 to num_counters-1.
var task_condition = 0;
//var task_condition = counterbalance;  // this indicated if people are in the CDFR or the DFR condition. For this experiment, 0 = DFR and 1 = CDFR. This only determined in the next function what happens after present_item runs.
// instructions for the recall period --- either free recall or serial recall
//var recall_instruction_condition = condition;  // co-opting the condition variable to use for the Exp4 recall instructions (whther the susprise mem test gives free or serial instructions

encoding_condition = 4

 // digits to use in constructing math distractor problems
var one_to_nine = [1, 2, 3, 4, 5, 6, 7, 8, 9];


// preallocate and initialize variables
var cur_list_num = 0; //counter to keep track of current list --- uses zero indexing
// trying starting out as cur_phase = 'STUDY' for the next next statement



/********************
* HTML manipulation
*
* All HTML files in the templates directory are requested 
* from the server when the PsiTurk object is created above. We
* need code to get those pages from the PsiTurk object and 
* insert them into the document.
*
********************/

// pick greeting page (first instruction page) based on instruction_condition
if (instruction_condition==0) {
    greeting = ["instructions/instructions-explicit-greeting.html"]
}
else if (instruction_condition==1) {
    greeting = ["instructions/instructions-incidental-greeting.html"]
}


/****** Will probably edit this task condition block too ******/
// pick task page (second instruction page) based on task_condition
if (encoding_condition==0) {
    task = "instructions/instructions-size-task.html"
    task_string = '<p>Is it easy to judge if it would it fit in a shoebox?</p>'
}
else if (encoding_condition==1) {
    task = "instructions/instructions-deepitem-task.html"
    task_string = '<p>Is it easy to generate a mental movie about this word?</p>'
}
else if (encoding_condition==2) {
    task = "instructions/instructions-deeprelational-task.html"
    task_string = '<p>Is it easy to incorporate this word in to your mental movie?</p>'
}
else if (encoding_condition==3) {
    task = "instructions/instructions-scenario-task.html"
    task_string = '<p>Is it easy to judge the relevance of this word to moving to a foreign land?</p>'
}
else if (encoding_condition==4) {
    task = "instructions/instructions-animacy-task.html"
    task_string = '<p>Is it easy to judge if this word refers to something that is alive?</p>'
}
else if (encoding_condition==5) {
    task = "instructions/instructions-weight-task.html"
    task_string = '<p>Is it easy to judge if it is heavier than a bottle of water?</p>'
}
else if (encoding_condition==6) {
    task = "instructions/instructions-home-task.html"
    task_string = '<p>Is it easy to judge if it would fit through your front door?</p>'
}
else if (encoding_condition==7) {
    task = "/instructions/instructions-E4constant-size-task.html"
    task_string = '<center><p>Is it easy to judge if it is larger than a <strong>shoebox</strong>?</p></center>'
}
else if (encoding_condition==8) {
    task = "/instructions/instructions-E4varying-size-task.html"
    // for each item, picks task string from the shuffled size_referents list we made above
}


// task_condition is constant or change
// recall_instruction_condition is free or serial
// do instruction first... need to pass


// pick emphasis page (third/last instruction page) based on instruction_condition
if (instruction_condition==0) {
    emph = "instructions/instructions-explicit-emphasis.html"
}
else if (instruction_condition==1) {
   emph = "instructions/instructions-incidental-emphasis.html"
}
var instructionPages = [greeting, task, emph];




// List of Task html pages
var pages = [
    "stage.html",
    "postquestionnaire.html",
];

// load all pages we will use
psiTurk.preloadPages(instructionPages);
psiTurk.preloadPages(pages);


/*
********************************
*******  FREE RECALL TASK ******
*/


var RunFR = function() {

    /******
     * Setup variables for this list
     *
     ****/
    var cur_phase, // trying tpo being with study, and then it will be changed later
        wordon, // time word is presented
        listening = false,
        first_recall = true, // keep tack of whether this is the first recall for a list
        distractor_done = false, // has the end of list distractor been finished yet
        stims = word_pool.splice(0,list_length); // get the items for this list: the next list_length elements of word pool
        size_tasks = size_referents.splice(0,list_length); // get the size referents for this list: the next list_length elements of word pool

  /******
     * Function to advance to the next item or recall depending on the phase of the list
     *
     ****/



        var next = function() {

        //figure out which phase of the task we are in

        // what to do if end of list distractor phase
        

        // what to do if recall phase
        if (stims.length === 0 && distractor_done) { // if there are no stims left and distractor is done, we have entered the recall phase
            if (first_recall) {
                //remove_word()
                //d3.select("#recall_input").remove(); this was a bad idea because it gets rid of it forever instead of just reassigning it, which will happen in recall_period()
                document.body.style.backgroundColor = "white";

                cur_phase = "RECALL"
                start_time = new Date().getTime();
            }
            recall_period()
        }

        else if (cur_phase === 'STUDY') { //if we just did a study phase, next is the distractor phase
            remove_word()
            // start if statement about here for if task_condition = CDFR or DFR
            if (task_condition == 1 || stims.length == 0) { //If CDFR condition or the final word was just presented for either CDFR or the DFR conditions, start distractor_task()
            	cur_phase = "DISTRACTOR";
            	distractor_start_time = new Date().getTime(); 
            	setTimeout(function(){wrapup_distractor(); }, distractor_delay); //start the distractor phase after a study phase
            	distractor_task();
            }

        }

        // what to do if study phase
        else { // otherwise we still have items to present
            cur_phase = "STUDY"
            stim = stims.shift();

            document.body.style.backgroundColor = "white";


            if (stims.length===list_length-1) {
                if (cur_list_num==0) {
                    ready_message = "The list will begin shortly. Position your fingers over the 'y' and 'n' keys so you are ready to respond!"
                }
                else {
                    ready_message = "A new list will begin shortly. Position your fingers over the 'y' and 'n' keys so you are ready to respond!"
                }
                d3.select("#stim")
                    .append("div")
                    .attr("id","word")
                    .style("color","black")
                    .style("text-align","center")
                    .style("font-size","40px")
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

    /*var next = function() {

     //figure out which phase of the task we are in

     // what to do if recall phase
     // else if (stims.length===0 && distractor_done) { // if there are no stims left, we have entered the recall phase
     if (stims.length !== 0) {

         if (cur_phase === 'STUDY') {
             if (stims.length === list_length - 1) {
                 if (cur_list_num === 0) {
                     ready_message = "The list will begin shortly. Position your fingers over the 'y' and 'n' keys so you are ready to respond!"
                 }
                 else if (cur_lis_num === 1) {
                     ready_message = "A new list will begin shortly. Position your fingers over the 'y' and 'n' keys so you are ready to respond!"
                 }
                 // Set HTML stuff
                 d3.select("#stim")
                     .append("div")
                     .attr("id", "word")
                     .style("color", "black")
                     .style("text-align", "center")
                     .style("font-size", "40px")
                     .style("font-weight", "400")
                     .style("margin", "20px")
                     .text(ready_message);
                 setTimeout(function () {
                     present_item;
                 }, delay_between_lists);
             }
             else {
                 remove_word()
                 distractor_task();
             }
         }
         else if (cur_phase === 'DISTRACTOR') {
             remove_word()
             present_item(stim[0]);
         }
     }
     else if (stims.length === 0) { // if there are no stims left, we have entered the recall phase {
         if (first_recall) {
             remove_word()
             cur_phase = "RECALL"
             start_time = new Date().getTime();
         }
         recall_period()
     }

 } */ 
  /****var next = function() {

        //figure out which phase of the task we are in
        // what to do if end of final distractor phase
        if (stims.length===0 && !distractor_done) { // if there are no stims left, we have entered the recall phase
            if (cur_phase = "DISTRACTOR"){
                remove_word()
                cur_phase = "STUDY";
                setTimeout(function(){wrapup_distractor(); }, distractor_delay); // start a timer that will end the distraction period
                distractor_task()
            }

        }

        // what to do if recall phase
        // else if (stims.length===0 && distractor_done) { // if there are no stims left, we have entered the recall phase
        if (stims.length===0) { // if there are no stims left, we have entered the recall phase
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
                    ready_message = "The list will begin shortly. Position your fingers over the 'y' and 'n' keys so you are ready to respond!"
                }
                else {
                    ready_message = "A new list will begin shortly. Position your fingers over the 'y' and 'n' keys so you are ready to respond!"
                }
                // Set HTML stuff
                d3.select("#stim")
                    .append("div")
                    .attr("id","word")
                    .style("color","black")
                    .style("text-align","center")
                    .style("font-size","40px")
                    .style("font-weight","400")
                    .style("margin","20px")
                    .text(ready_message);
                setTimeout(function(){present_item( stim[0] ); }, delay_between_lists);
            }
            else{
                remove_word()
                distractor_task();
                }

        }

    };
****/


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
//        finish() // useful for debugging --- will automatically end the experiment unpon button press if the handler is listening

        var keyCode = e.keyCode,
            subjects_answer,
            recalled_item,
            response;


         /***** Will need something like the study handler in the distractor function *****/
        //!!!!!!!!!!!!!!!!!! HANDLER for the study phase
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
                    d3.select("#query").html('<p id="prompt"> <span style="color: red; "><center>Invalid Response. Press Y or N</span></p></center>');
                    break;
            }



            if (response.length > 0) {
                listening = false;
                var rt = new Date().getTime() - wordon;
                // remove the task prompts (turn them white) to give participant a subtle cue that the response was detected
                d3.select("#query").html('<p id="prompt"> <center> <span style="color: black; ">Thanks for responding!</span></p>');
                d3.select("#task").html('<p id="prompt"> <center> <span style="color: black; ">Thanks for responding!</span></p>');

                psiTurk.recordTrialData({
                        'instruction_condition': instruction_condition,
                        //'recall_instruction_condition': recall_instruction_condition,
                        'task_condition': task_condition,
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

            // only accept input if the recall text box is in focus
            box_in_focus = document.activeElement.name == "recall_field";
            if (!box_in_focus)  {
                d3.select("#recall_input").html('<span  style="color: red;">Click inside textbox to activate it before typing!:</span> ' +
                    '<input type="text" id="recall_field" name="recall_field"/>');
                    return
            }

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
                        'instruction_condition': instruction_condition,
                        //'recall_instruction_condition': recall_instruction_condition,
                        'task_condition': task_condition,
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

            // only accept input if the input text box is in focus
            box_in_focus = document.activeElement.name == "recall_field";
            if (!box_in_focus)  {
                d3.select("#recall_input").html('<span  style="color: red;">Click inside textbox to activate it before typing!:</span> ' +
                    '<input type="text" id="recall_field" name="recall_field"/>');
                    return
            }

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

                var elapsed = new Date().getTime() - distractor_start_time;
                psiTurk.recordTrialData({
                        'instruction_condition': instruction_condition,
                        //'recall_instruction_condition': recall_instruction_condition,
                        'task_condition': task_condition,
                        'list': cur_list_num,
                        'phase': "DISTRACTOR",
                        'correct_answer': correct_answer,
                        'subjects_answer': subjects_answer,
                        'rt': elapsed
                    }
                );
                if (elapsed < distractor_delay) {
                    distractor_task();
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

        // remove any words already on screen
        remove_word()

        // show the word for word_pres_rate ms
        if (encoding_condition==4) {
            d3.select("#task").html(task_string);
        }
        else if (task_condition==8) {
            cur_task = size_tasks.shift()
            d3.select("#task").html(cur_task[0]);
        }
        d3.select("#query").html('<p id="prompt"><center>press "Y" for yes, "N" for no.</center></p>');
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
        cur_phase = 'STUDY'

        setTimeout(function(){wrapup_word(); }, word_pres_rate);

    };

       /******
     * Function to present text box for recalling words
     *
     ****/
    var recall_period = function() {
        //remove text from encoding task
        d3.select("#query").remove();
 //     d3.select("#recall_input").remove();

        // display input box
        if (instruction_condition==0) {
            if (encoding_condition==4) {
                disp_this = '<p>You now have ' + recall_time/1000 +
                ' seconds to to try and recall the words from the list you just saw. ' +
                'You can recall the words in any order. Try to recall as many words as you can. If you cannot remember any more words, that is okay; the task will automatically advance when the time is up.</p>'
            }
            /*else if (task_condition==8) {
                disp_this = '<p>You now have ' + recall_time/1000 +
                ' seconds to to try and recall the words from the list you just saw. Recall the words that were presented in large font in the center of the screen, <strong>NOT</strong> the size referents that appared in smaller font above the words. ' +
                'You can recall the words in any order. Try to recall as many words as you can. If you cannot remember any more words, that is okay; the task will automatically advance when the time is up.</p>'
            }*/

        }
        else if (instruction_condition==1) {
            if (encoidng_condition==4) {
                disp_this = '<p>You now have ' + recall_time / 1000 +
                ' seconds to to try and recall the words from the list you just saw. Try to recall the words in the <strong>same order you saw them</strong>' +
                '. Try to recall as many words as you can. If you cannot remember any more words, that is okay; the task will automatically advance when the time is up.</p>'
            }
            /*else if (task_condition==8) {
                disp_this = '<p>You now have ' + recall_time / 1000 +
                ' seconds to to try and recall the words from the list you just saw. Recall the words that were presented in large font in the center of the screen, <strong>NOT</strong> the size referents that appared in smaller font above the words. Try to recall the words in the <strong>same order you saw them</strong>' +
                '. Try to recall as many words as you can. If you cannot remember any more words, that is okay; the task will automatically advance when the time is up.</p>'
            }*/
        }





        d3.select("#task").html(disp_this);
        d3.select("#recall_input").html('<span>Type a word and press ENTER to submit:</span> ' +
            '<input type="text" id="recall_field" name="recall_field"/>');
        d3.select("#stim").html('<span>  <span>')

        // want to ensure people don't accidently try to enter their last response from the math distractor here
        // so defocus the textbox if this is the first recall to avoid too rapid a response
        if (!first_recall) {
            d3.select("#recall_field").node().focus()
        }


        // start listening
        listening = true;

        if (first_recall) {
            document.body.style.backgroundColor = "white";
            setTimeout(function(){wrapup_recall(); }, recall_time);
            setTimeout(function(){d3.select("#recall_field").node().focus(); }, recall_box_lag/2);
            first_recall = false
        }

    };

    //runs math distractor
   var distractor_task = function() {
        //remove text from encoding task
        //d3.select("#query").remove();
      //  d3.select("#task").remove();

        // setup math problem
        shuffled_digits = _.shuffle(one_to_nine);
        A = shuffled_digits[0];
        B = shuffled_digits[1];
        C = shuffled_digits[2];
        correct_answer = A + B + C;
        subjects_answer = [];

        // display task text
        disp_this = '<p>You will now solve math problems for ' + distractor_delay/1000 +
            ' seconds. Try to solve as many problems as you can without sacrificing accuracy. The task will automatically advance when the time is up.</p>'
        //disp_this = '<p><center>Judge whether or not the following problem is correct:</center></p>'
        d3.select("#task").html(disp_this);
        //d3.select("#task").html(disp_this);
        problem = '<p><span style="color: black; font-size: 50px">' + A + '+' + B + '+' + C + '=?</span></p>'
       //problem = '<p><center><span style="color: black; font-size: 50px">' + A + '+' + B + '+' + C + '=' + correct_answer + '</center></span></p>'
        d3.select("#query").html(problem);
        //d3.select("#word").html(problem)
        d3.select("#recall_input").html('<span>Add the three numbers, type your answer, and press ENTER to submit:</span> ' +
            '<input type="text" id="recall_field" name="recall_field"/>');
        //d3.select("#query").html('<p id="prompt"><center>press "Y" for yes, "N" for no.</center></p>');

       //setTimeout(function(){wrapup_distractor(); }, word_pres_rate);

        // display input box
        d3.select("#recall_field").node().focus()
        cur_phase = "DISTRACTOR";
        listening = true;           


        //d3.select("#stim").remove()

                //return problem;

        //cur_phase = "DISTRACTOR";
        //listening = true;
        //distractor_start_time = new Date().getTime();


        // display task text    
        //disp_this = '<p><center>Judge whether or not the following problem is correct:</center></p>'
        //d3.select("#task").html(disp_this);
        //problem = '<p><center><span style="color: black; font-size: 50px">' + A + '+' + B + '+' + C + '=' + correct_answer + '</center></span></p>'
        //d3.select("#word").html(problem)
        //d3.select("#query").html('<p id="prompt"><center>press "Y" for yes, "N" for no.</center></p>');
        //setTimeout(function(){wrapup_distractor(); }, distractor_pres_rate);

        //return problem;

        //cur_phase = "DISTRACTOR";
        //listening = true;
        //distractor_start_time = new Date().getTime();

    }

    var wrapup_distractor = function() {
       cur_phase = "DISTRACTOR";
       if (stims.length === 0) {
            distractor_done = true;
        }
        document.body.style.backgroundColor = "red";
        //trying to reassign #recall_input
        d3.select("#recall_input").html('<span>    </span> ');
        if (listening) {
            var rt = new Date().getTime() - distractor_start_time;
            psiTurk.recordTrialData({
                    'instruction_condition': instruction_condition,
                    //'recall_instruction_condition': recall_instruction_condition,
                    'task_condition': task_condition,
                    'list': cur_list_num,
                    'phase': "DISTRACTOR",
                    'response': "Timed_Out",
                    'rt': rt
                }
            );
        }

        // show a fixation for isi ms
        // remove text from encoding task
        // stop listening
        listening = false;
        //d3.select("#word").remove();
        d3.select("#query").html('<p id="prompt"> <center> <span style="color: red; ">Thanks for responding!</span></p>');
        d3.select("#task").html('<p id="prompt"> <center> <span style="color: red; ">Thanks for responding!</span></p>');

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
     * Function to record a time out response if no response was made to the word and move on
     *
     ****/
     var wrapup_word = function() {

        // record a time out if they have not responded to the word
        if (listening) {
            var rt = new Date().getTime() - wordon;
            psiTurk.recordTrialData({
                    'instruction_condition': instruction_condition,
                    //'recall_instruction_condition': recall_instruction_condition,
                    'task_condition': task_condition,
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
        d3.select("#query").html('<p id="prompt"> <center> <span style="color: black; ">Thanks for responding!</span></p>');
        d3.select("#task").html('<p id="prompt"> <center> <span style="color: black; ">Thanks for responding!</span></p>');

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

        // record strategy questions:
        $('input').each( function(i, val) {
            psiTurk.recordTrialData({
                        'instruction_condition': instruction_condition,
                        'task_condition': task_condition,
                        //'recall_instruction_condition': recall_instruction_condition,
                        'phase': "postquestionnaire",
                        'strategy': this.name,
                        'strategy_description': this.value,
                        'used': this.checked,
                        });

        });

        // record awarness question
        $('select').each( function(i, val) {
            psiTurk.recordTrialData({
                        'instruction_condition': instruction_condition,
                        'task_condition': task_condition,
                        //'recall_instruction_condition': recall_instruction_condition,
                        'phase': "postquestionnaire",
                        'aware_question': this.name,
                        'aware_ans': this.value,
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

    psiTurk.showPage('debriefing.html');
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










