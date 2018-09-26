const input = 'Hello there, my large undersea friend'; //What you want to say
const vowels = ['a', 'e', 'i', 'o', 'u']; //Whales speak only in vowels
const resultArray = [ ];

//Compare every letter in input to all of the vowels and store translation in resultArray
for (let inputIndex = 0; inputIndex < input.length; inputIndex++) {
  for (let vowelIndex = 0; vowelIndex < vowels.length; vowelIndex++) {
  if (input[inputIndex] === vowels[vowelIndex]) {
    resultArray.push(input[inputIndex]);
  }
 }
  if (input[inputIndex] === 'e') { //Whales have double 'e's
    resultArray.push(input[inputIndex]);
  }
  if (input[inputIndex] === 'u') { ////Whales have double 'e's
      resultArray.push(input[inputIndex]);
  }
};

console.log(resultArray.join('').toUpperCase()); //Whale song is loud
//Deliver translation