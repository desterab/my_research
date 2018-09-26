//creating a parent class Media
class Media {
  constructor(title) {
    this._title = title;
    this._isCheckedOut = false;
    this._ratings = [];
  }

  //creating a getters for tittle, isCheckedOut and ratings
  get title() {
    return this._title;
  }
  get isCheckedOut () {
    return this._isCheckedOut;
  }
  get ratings() {
    return this._ratings;
  }

//creating a method toggleCheckOutStatus that changes the values saved to the _isCheckedOut property
  toggleCheckOutStatus() {
    this._isCheckedOut = !this._isCheckedOut;
  }
//creating a method getAverageRating that returns an average value of ratings array
  getAverageRating () {
    let ratingsSum = this.ratings.reduce((currentSum, rating) => currentSum + rating, 0);
    let averageRating = ratingsSum / this.ratings.length
    return averageRating;
  }
//creating a method addRating
  addRating(ratings) {
    if (ratings > 5 || ratings < 1) {
      console.log('Please enter a rating 1-5.');
    } else {
      this._ratings.push(ratings);
  }
}
}

//creating a Book class that extends Media/parent class
class Book extends Media {
  constructor(title, author, pages) {
    super(title);
    this._author = author;
    this._pages = pages;
  }

  get author() {
    return this._title;
  }
  get pages() {
    return this.pages;
  }
}



//create a new book instance
const historyOfEverything = new Book ('A Short History of Nearly Everything', 'Bill Bryson', 544);

console.log(historyOfEverything.isCheckedOut);
historyOfEverything.toggleCheckOutStatus();
console.log(historyOfEverything.isCheckedOut);

historyOfEverything.addRating(5);
historyOfEverything.addRating(5);
historyOfEverything.addRating(5);
historyOfEverything.addRating(1);

console.log(historyOfEverything.getAverageRating());

//creating a Movie class that extends Media/parent class
class Movie extends Media {
  constructor(title, director, runTime) {
    super(title);
    this._director = director;
    this._runTime = runTime;
  }
  get director () {
    return this._director;
  }
  get runTime () {
    return this._runTime;
  }
}


//create a new movie instance
const speed = new Movie ('Speed', 'Jan de Bont', 116);
speed.toggleCheckOutStatus();
console.log(speed._isCheckedOut);

speed.addRating(1);
speed.addRating(1);
speed.addRating(5);

console.log(speed.getAverageRating());

//creating a CD class that extends Media/parent class
class CD extends Media {
  constructor (title, artist, songs) {
    super(title);
    this._artist = artist;
    this._songs = [];
  }

  get artist() {
    return this._artist;
  }
  get songs() {
    return this._songs
  }
  //todo: try to create shuffle method
}

//creating a new CD instance
const badBlood = new CD ('Bad Blood', 'Bastille', ["Pompeii", "Things We Lost in the Fire", "Bad Blood", "Overjoyed", "These Streets", "Weight of Living, Pt. II", "Icarus", "Oblivion", "Flaws", "Daniel in the Den", "Laura Palmer", "Get Home", "Weight of Living, Pt. I", "Laughter Lines"]);
badBlood.toggleCheckOutStatus();
console.log(badBlood._isCheckedOut);

badBlood.addRating(3);
badBlood.addRating(5);
badBlood.addRating(5);
badBlood.addRating(5);

console.log(badBlood.getAverageRating());























