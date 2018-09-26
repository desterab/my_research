const menu = {
  _courses: {
    _appetizers: [],
    _mains: [],
    _desserts: [],

  	get appetizers() {
    return this._appetizers;
  	},
  	set appetizers(appetizerIn) {
    this._appetizers = appetizersIn;
  	},
  	get mains() {
    return this._mains;
  	},
  	set mains(mainIn) {
    this._mains = mainsIn;
 		},
  	get desserts() {
    return this._desserts;
  	},
  	set desserts(dessertIn) {
    this._desserts = dessertsIn;
  	},
  },

 get courses() {
   return {
     appetizers: this._courses.appetizers, //this uses the appetizer getter method
     mains: this._courses.mains, //this uses the mains getter method
     desserts: this._courses.desserts, //this uses the dessert getter method
   };
 	},
  ///======================================================
	addDishToCourse (courseName, dishName, dishPrice ) {
  const dish = {
  	name: dishName,
    price: dishPrice
	};

	this._courses[courseName].push(dish);
	},

  getRandomDishFromCourse: function (courseName) {
    const dishes = this._courses[courseName];
    const randomIndex = Math.floor(Math.random() * dishes.length);
    return dishes [randomIndex]
  },
	generateRandomMeal: function () {
    const appetizer = this.getRandomDishFromCourse('appetizers');
    const mains = this.getRandomDishFromCourse('mains');
    const desserts = this.getRandomDishFromCourse('desserts');
    const totalPrice = appetizer.price + mains.price + desserts.price

    return `You will be eating ${appetizer.name}, followed by a lovely ${mains.name} and ${desserts.name}. The price is ${totalPrice}.`;
  }
};

menu.addDishToCourse('appetizers', 'Cesar Salad', 4.25);
menu.addDishToCourse('appetizers', 'Prawn Coctail', 4.25);
menu.addDishToCourse('appetizers', 'Garlic Bread', 3.50);

menu.addDishToCourse('mains', 'Lasagna', 9.75);
menu.addDishToCourse('mains', 'Ribeye Steak', 14.95);
menu.addDishToCourse('mains', 'Fish & Chips', 12.95);

menu.addDishToCourse('desserts', 'Cheese Cake', 4.50);
menu.addDishToCourse('desserts', 'Creme Brule', 4.25);
menu.addDishToCourse('desserts', 'Cheese Board', 3.25);

let meal = menu.generateRandomMeal();

console.log(meal)