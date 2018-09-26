const team = {
  _players: [
    {firstName: 'Pablo', lastName: 'Sanchez', age: 11},
  	{firstName: 'Selena', lastName: 'Perian', age: 12},
    {firstName: 'Jay', lastName: 'Schwartzenagger', age: 12}
  ],
  _games: [
    {opponent: 'Seals', teamPoints: 12, opponentPoints: 6},
    {opponent: 'Great Clips', teamPoints: 7, opponentPoints: 6},
    {opponent: 'Oranges', teamPoints: 2, opponentPoints: 5}
  ],

  get games() {
    return this._games
  },

  addPlayer(firstName, lastName, age) {
  	let player =  {
  		firstName: firstName,
  		lastName: lastName,
  		age: age
		};

		this._players.push(player)
	},
  addGame(opponent, teamPoints, opponentPoints) {
    let game = {
      opponent: opponent,
      teamPoints: teamPoints,
      opponentPoints: opponentPoints,
    };

    this._games.push(game)
  }
};

team.addPlayer('Steph', 'Curry', 8);
team.addPlayer('Lisa', 'Leslie', 10);
team.addPlayer('Bugs', 'Bunny', 9);

team.addGame('Jazz Hands', 15, 12)
team.addGame('Alligators', 14, 2)
team.addGame('X-Factor', 15, 20)

console.log(team._players)
console.log(team._games)