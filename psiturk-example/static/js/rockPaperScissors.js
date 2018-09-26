
const getUserChoice = (userInput) => {
  userInput = userInput.toLowerCase();
  if (userInput === 'rock' || userInput === 'paper' || userInput === 'scissors' ) {
    return userInput;
  } else if (userInput === 'bomb') {
    return userInput
  } else {
    return 'Error'
    console.log('Error!');
  }
}

const getComputerChoice = () => {
  let number = Math.floor(Math.random() * 3);
  switch (number) {
    case 0: return 'rock';
      break;
    case 1: return 'paper';
      break;
    case 2: return 'scissors';
      break;
  }
};

const determineWinner = (userChoice, computerChoice) => {
  if (userChoice === computerChoice) {
    return 'The game is a tie.';
  }
  if (userChoice == 'Error') {
    return 'You did not throw anything, so you lost. Boo hoo.'
  }
  if (userChoice === 'rock') {
    if (computerChoice === 'paper') {
      return 'The computer wins.';
    } else {
      return 'You win!!!';
    }
  }
  if (userChoice === 'paper') {
     if (computerChoice === 'scissors') {
       return 'The computer wins';
     } else {
       return 'You win!!!';
     }
   }
  if (userChoice === 'scissors') {
    if (computerChoice === 'rock') {
      return 'The computer wins.';
    } else {
      return 'You win!!!';
    }
  }
 	if (userChoice == 'bomb') {
    return 'You win!!!'
  }
};

const playGame = () => {
  const userChoice = getUserChoice('bomb');
  const computerChoice = getComputerChoice();
  console.log(`You threw ${userChoice}`);
  console.log(`The computer threw ${computerChoice}`);

  console.log(determineWinner(userChoice, computerChoice));
};


playGame();