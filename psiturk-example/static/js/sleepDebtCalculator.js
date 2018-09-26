// Change return values here to reflect how much actual sleep was attained in the week
const getSleepHours = day => {
  switch (day) {
    case 'monday':
      return 15;
      break;
    case 'tuesday':
      return 7;
      break;
    case 'wednesday':
      return 8;
      break;
    case 'thursday':
      return 7;
      break;
    case 'friday':
      return 8;
      break;
    case 'saturday':
      return 7;
      break;
    case 'sunday':
      return 6;
      break;
  }
}

// This adds up the hours of sleep for the week
const getActualSleepHours = () => getSleepHours('monday') +
  getSleepHours('tuesday') +
  getSleepHours('wednesday') +
  getSleepHours('thursday') +
  getSleepHours('friday') +
  getSleepHours('saturday') +
  getSleepHours('sunday');

// This calculates the ideal number of hours of sleep in a week. Change the value for idealHours to match your ideal hours.
const getIdealSleepHours = () => {
  var idealHours = 8.5;
  return idealHours*7;
};

// This calculates if someone has sleep debt and how much.
const calculateSleepDebt = () => {
  let actualSleepHours = getActualSleepHours();
  let idealSleepHours = getIdealSleepHours();
  if (actualSleepHours === idealSleepHours) {
    console.log('You get the perfect amount of sleep');
  } else if (actualSleepHours < idealSleepHours) {
    let calculateSleepDebt = idealSleepHours - actualSleepHours;
    console.log(`You get ${calculateSleepDebt} hours of sleep less than you need. You need some rest.`);
  } else if (actualSleepHours > idealSleepHours) {
      let calculateSleepExcess = actualSleepHours - idealSleepHours;
      console.log(`You get ${calculateSleepExcess} hours of sleep more than you need.`);
  }
}
// Run the program.
calculateSleepDebt()