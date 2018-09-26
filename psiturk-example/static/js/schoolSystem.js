//create parent class School
class School {
  constructor(name, level, numberOfStudents) {
    this._name = name;
    this._level = level;
    this._numberOfStudents = numberOfStudents;
  }
  //get keys
  get name() {
    return this._name;
  }
  get level() {
    return this._level;
  }
  get numberOfStudents() {
    return this._numberOfStudents;
  }

  quickFacts() {
    console.log(`${this._name} educates ${this._numberOfStudents} at the ${this._level} level.`);
  }
  static pickSubstituteTeacher(substituteTeachers) {
    const randomIndex = Math.floor(substituteTeachers.length * Math.random());
    let teacher = substituteTeachers[randomIndex];
    return teacher;
  }

  //setter for numberOfStudents
  set numberOfStudents(numberOfStudentsInput) {
    if (typeof numberOfStudentsInput == 'number') {
      this._numberofStudents = numberOfStudentsIn;
    }
    else {
      console.log('Invalid input: numberOfStudents must be set to a number');
      return 'Invalid input';
    };
  }
}

//create class primary school
class PrimarySchool extends School {
  constructor(name, numberOfStudents, pickupPolicy) {
    super(name, 'primary', numberOfStudents);
    this._pickupPolicy = pickupPolicy;
  }
  get pickupPolicy() {
    return this._pickupPolicy;
  }
}

//create Primary School instance
const lorraineHansbury = new PrimarySchool ('Lorraine Hansbury', 514, 'Students must be picked up by a parent, guardian, or a family member over the age of 13.');

console.log(lorraineHansbury.quickFacts());

//get sub for Lorraine Hansbury school
let sub = School.pickSubstituteTeacher(['Jamal Crawford', 'Lou Williams', 'J. R. Smith', 'James Harden', 'Jason Terry', 'Manu Ginobli']);
console.log(sub);


//create class high school
class HighSchool extends School{
  constructor(name, numberOfStudents,sportsTeams) {
    super(name, 'high', numberOfStudents);
    this._sportsTeams = sportsTeams;
  }
  get sportsTeams() {
    return this._sportsTeams;
  }
}

//create HS instance
const alSmith = new HighSchool ('Al E. Smith', 415, ['Baseball', 'Basketball', 'Volleyball', 'Track and Field']);

console.log(alSmith.quickFacts());
console.log(alSmith._sportsTeams)

//I DONT KNOW WHY IT ALSO RETURNS UNDEFINED FOR EVERY QUICKFACTS