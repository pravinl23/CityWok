// South Park canonical episode titles mapping
// Key: (season, episode) -> title

const EP_TITLE = {
  // Season 1 (1997–1998)
  "1,1": "Cartman Gets an Anal Probe",
  "1,2": "Volcano",
  "1,3": "Weight Gain 4000",
  "1,4": "Big Gay Al's Big Gay Boat Ride",
  "1,5": "An Elephant Makes Love to a Pig",
  "1,6": "Death",
  "1,7": "Pinkeye",
  "1,8": "Starvin' Marvin",
  "1,9": "Mr. Hankey, the Christmas Poo",
  "1,10": "Damien",
  "1,11": "Tom's Rhinoplasty",
  "1,12": "Mecha-Streisand",
  "1,13": "Cartman's Mom Is a Dirty Slut",

  // Season 2 (1998–1999)
  "2,1": "Terrance and Phillip in Not Without My Anus",
  "2,2": "Cartman's Mom Is Still a Dirty Slut",
  "2,3": "Ike's Wee Wee",
  "2,4": "Chickenlover",
  "2,5": "Conjoined Fetus Lady",
  "2,6": "The Mexican Staring Frog of Southern Sri Lanka",
  "2,7": "City on the Edge of Forever",
  "2,8": "Summer Sucks",
  "2,9": "Chef's Chocolate Salty Balls",
  "2,10": "Chickenpox",
  "2,11": "Roger Ebert Should Lay Off the Fatty Foods",
  "2,12": "Clubhouses",
  "2,13": "Cow Days",
  "2,14": "Chef Aid",
  "2,15": "Spookyfish",
  "2,16": "Merry Christmas, Charlie Manson!",
  "2,17": "Gnomes",
  "2,18": "Prehistoric Ice Man",

  // Season 3 (1999–2000)
  "3,1": "Rainforest Shmainforest",
  "3,2": "Spontaneous Combustion",
  "3,3": "Succubus",
  "3,4": "Tweek vs. Craig", // switched (originally 3,5)
  "3,5": "Jakovasaurs", // switched (originally 3,4)
  "3,6": "Sexual Harassment Panda",
  "3,7": "Cat Orgy",
  "3,8": "Two Guys Naked in a Hot Tub",
  "3,9": "Jewbilee",
  "3,10": "Chinpokomon", // swithced (originally 3,11)
  "3,11": "Starvin' Marvin in Space", // switched (originally 3,13)
  "3,12": "Korn's Groovy Pirate Ghost Mystery", // switched (originally 3,10)
  "3,13": "Hooked on Monkey Fonics", // switched (originally 3,12)
  "3,14": "The Red Badge of Gayness",
  "3,15": "Mr. Hankey's Christmas Classics",
  "3,16": "Are You There God? It's Me, Jesus",
  "3,17": "World Wide Recorder Concert",

  // Season 4 (2000)
  "4,1": "The Tooth Fairy's Tats 2000",
  "4,2": "Cartman's Silly Hate Crime 2000",
  "4,3": "Timmy 2000",
  "4,4": "Quintuplets 2000",
  "4,5": "Cartman Joins NAMBLA",
  "4,6": "Cherokee Hair Tampons",
  "4,7": "Chef Goes Nanners",
  "4,8": "Something You Can Do with Your Finger",
  "4,9": "Do the Handicapped Go to Hell?",
  "4,10": "Probably",
  "4,11": "Fourth Grade",
  "4,12": "Trapper Keeper",
  "4,13": "Helen Keller! The Musical",
  "4,14": "Pip",
  "4,15": "Fat Camp",
  "4,16": "The Wacky Molestation Adventure",
  "4,17": "A Very Crappy Christmas",

  // Season 5 (2001)
  "5,1": "It Hits the Fan",
  "5,2": "Cripple Fight",
  "5,3": "Super Best Friends",
  "5,4": "Scott Tenorman Must Die",
  "5,5": "Terrance and Phillip: Behind the Blow",
  "5,6": "Cartmanland",
  "5,7": "Proper Condom Use",
  "5,8": "Towelie",
  "5,9": "Osama bin Laden Has Farty Pants",
  "5,10": "How to Eat with Your Butt",
  "5,11": "The Entity",
  "5,12": "Here Comes the Neighborhood",
  "5,13": "Kenny Dies",
  "5,14": "Butters' Very Own Episode",

  // Season 6 (2002)
  "6,1": "Jared Has Aides",
  "6,2": "Asspen",
  "6,3": "Freak Strike",
  "6,4": "Fun with Veal",
  "6,5": "The New Terrance and Phillip Movie Trailer",
  "6,6": "Professor Chaos",
  "6,7": "Simpsons Already Did It",
  "6,8": "Red Hot Catholic Love",
  "6,9": "Free Hat",
  "6,10": "Bebe's Boobs Destroy Society",
  "6,11": "Child Abduction Is Not Funny",
  "6,12": "A Ladder to Heaven",
  "6,13": "The Return of the Fellowship of the Ring to the Two Towers",
  "6,14": "The Death Camp of Tolerance",
  "6,15": "The Biggest Douche in the Universe",
  "6,16": "My Future Self 'n' Me",
  "6,17": "Red Sleigh Down",

  // Season 7 (2003)
  "7,1": "Cancelled",
  "7,2": "Krazy Kripples",
  "7,3": "Toilet Paper",
  "7,4": "I'm a Little Bit Country",
  "7,5": "Fat Butt and Pancake Head",
  "7,6": "Lil' Crime Stoppers",
  "7,7": "Red Man's Greed",
  "7,8": "South Park Is Gay",
  "7,9": "Christian Rock Hard",
  "7,10": "Grey Dawn",
  "7,11": "Casa Bonita",
  "7,12": "All About Mormons",
  "7,13": "Butt Out",
  "7,14": "Raisins",
  "7,15": "It's Christmas in Canada",

  // Season 8 (2004)
  "8,1": "Good Times with Weapons",
  "8,2": "Up the Down Steroid",
  "8,3": "The Passion of the Jew",
  "8,4": "You Got F'd in the A",
  "8,5": "Awesom-O",
  "8,6": "The Jeffersons",
  "8,7": "Goobacks",
  "8,8": "Douche and Turd",
  "8,9": "Something Wall-Mart This Way Comes",
  "8,10": "Pre-School",
  "8,11": "Quest for Ratings",
  "8,12": "Stupid Spoiled Whore Video Playset",
  "8,13": "Cartman's Incredible Gift",
  "8,14": "Woodland Critter Christmas",

  // Season 9 (2005)
  "9,1": "Mr. Garrison's Fancy New Vagina",
  "9,2": "Die Hippie, Die",
  "9,3": "Wing",
  "9,4": "Best Friends Forever",
  "9,5": "The Losing Edge",
  "9,6": "The Death of Eric Cartman",
  "9,7": "Erection Day",
  "9,8": "Two Days Before the Day After Tomorrow",
  "9,9": "Marjorine",
  "9,10": "Follow That Egg!",
  "9,11": "Ginger Kids",
  "9,12": "Trapped in the Closet",
  "9,13": "Free Willzyx",
  "9,14": "Bloody Mary",

  // Season 10 (2006)
  "10,1": "The Return of Chef",
  "10,2": "Smug Alert!",
  "10,3": "Cartoon Wars Part I",
  "10,4": "Cartoon Wars Part II",
  "10,5": "A Million Little Fibers",
  "10,6": "ManBearPig",
  "10,7": "Tsst",
  "10,8": "Make Love, Not Warcraft",
  "10,9": "Mystery of the Urinal Deuce",
  "10,10": "Miss Teacher Bangs a Boy",
  "10,11": "Hell on Earth 2006",
  "10,12": "Go God Go",
  "10,13": "Go God Go XII",
  "10,14": "Stanley's Cup",

  // Season 11 (2007)
  "11,1": "With Apologies to Jesse Jackson",
  "11,2": "Cartman Sucks",
  "11,3": "Lice Capades",
  "11,4": "The Snuke",
  "11,5": "Fantastic Easter Special",
  "11,6": "D-Yikes!",
  "11,7": "Night of the Living Homeless",
  "11,8": "Le Petit Tourette",
  "11,9": "More Crap",
  "11,10": "Imaginationland Episode I",
  "11,11": "Imaginationland Episode II",
  "11,12": "Imaginationland Episode III",
  "11,13": "Guitar Queer-O",
  "11,14": "The List",

  // Season 12 (2008)
  "12,1": "Tonsil Trouble",
  "12,2": "Britney's New Look",
  "12,3": "Major Boobage",
  "12,4": "Canada on Strike",
  "12,5": "Eek, a Penis!",
  "12,6": "Over Logging",
  "12,7": "Super Fun Time",
  "12,8": "The China Probrem",
  "12,9": "Breast Cancer Show Ever",
  "12,10": "Pandemic",
  "12,11": "Pandemic 2: The Startling",
  "12,12": "About Last Night...",
  "12,13": "Elementary School Musical",
  "12,14": "The Ungroundable",

  // Season 13 (2009)
  "13,1": "The Ring",
  "13,2": "The Coon",
  "13,3": "Margaritaville",
  "13,4": "Eat, Pray, Queef",
  "13,5": "Fishsticks",
  "13,6": "Pinewood Derby",
  "13,7": "Fatbeard",
  "13,8": "Dead Celebrities",
  "13,9": "Butters' Bottom Bitch",
  "13,10": "W.T.F.",
  "13,11": "Whale Whores",
  "13,12": "The F Word",
  "13,13": "Dances with Smurfs",
  "13,14": "Pee",

  // Season 14 (2010)
  "14,1": "Sexual Healing",
  "14,2": "The Tale of Scrotie McBoogerballs",
  "14,3": "Medicinal Fried Chicken",
  "14,4": "You Have 0 Friends",
  "14,5": "200",
  "14,6": "201",
  "14,7": "Crippled Summer",
  "14,8": "Poor and Stupid",
  "14,9": "It's a Jersey Thing",
  "14,10": "Insheeption",
  "14,11": "Coon 2: Hindsight",
  "14,12": "Mysterion Rises",
  "14,13": "Coon vs. Coon and Friends",
  "14,14": "Crème Fraîche",

  // Season 15 (2011)
  "15,1": "HumancentiPad",
  "15,2": "Funnybot",
  "15,3": "Royal Pudding",
  "15,4": "T.M.I.",
  "15,5": "Crack Baby Athletic Association",
  "15,6": "City Sushi",
  "15,7": "You're Getting Old",
  "15,8": "Ass Burgers",
  "15,9": "The Last of the Meheecans",
  "15,10": "Bass to Mouth",
  "15,11": "Broadway Bro Down",
  "15,12": "1%",
  "15,13": "A History Channel Thanksgiving",
  "15,14": "The Poor Kid",

  // Season 16 (2012)
  "16,1": "Reverse Cowgirl",
  "16,2": "Cash for Gold",
  "16,3": "Faith Hilling",
  "16,4": "Jewpacabra",
  "16,5": "Butterballs",
  "16,6": "I Should Have Never Gone Ziplining",
  "16,7": "Cartman Finds Love",
  "16,8": "Sarcastaball",
  "16,9": "Raising the Bar",
  "16,10": "Insecurity",
  "16,11": "Going Native",
  "16,12": "A Nightmare on FaceTime",
  "16,13": "A Scause for Applause",
  "16,14": "Obama Wins!",

  // Season 17 (2013)
  "17,1": "Let Go, Let Gov",
  "17,2": "Informative Murder Porn",
  "17,3": "World War Zimmerman",
  "17,4": "Goth Kids 3: Dawn of the Posers",
  "17,5": "Taming Strange",
  "17,6": "Ginger Cow",
  "17,7": "Black Friday",
  "17,8": "A Song of Ass and Fire",
  "17,9": "Titties and Dragons",
  "17,10": "The Hobbit",

  // Season 18 (2014)
  "18,1": "Go Fund Yourself",
  "18,2": "Gluten Free Ebola",
  "18,3": "The Cissy",
  "18,4": "Handicar",
  "18,5": "The Magic Bush",
  "18,6": "Freemium Isn't Free",
  "18,7": "Grounded Vindaloop",
  "18,8": "Cock Magic",
  "18,9": "#REHASH",
  "18,10": "#HappyHolograms",

  // Season 19 (2015)
  "19,1": "Stunning and Brave",
  "19,2": "Where My Country Gone?",
  "19,3": "The City Part of Town",
  "19,4": "You're Not Yelping",
  "19,5": "Safe Space",
  "19,6": "Tweek x Craig",
  "19,7": "Naughty Ninjas",
  "19,8": "Sponsored Content",
  "19,9": "Truth and Advertising",
  "19,10": "PC Principal Final Justice",

  // Season 20 (2016)
  "20,1": "Member Berries",
  "20,2": "Skank Hunt",
  "20,3": "The Damned",
  "20,4": "Wieners Out",
  "20,5": "Douche and a Danish",
  "20,6": "Fort Collins",
  "20,7": "Oh, Jeez",
  "20,8": "Members Only",
  "20,9": "Not Funny",
  "20,10": "The End of Serialization as We Know It",
};

/**
 * Get episode title from episode ID
 * @param {string} episodeId - Episode ID in format "S03E15" or "S3E15"
 * @returns {string|null} Episode title or null if not found
 */
export function getEpisodeTitle(episodeId) {
  if (!episodeId) return null;
  
  // Parse episode ID (e.g., "S03E15" or "S3E15" -> season=3, episode=15)
  const match = episodeId.match(/S(\d+)E(\d+)/i);
  if (!match) return null;
  
  const season = parseInt(match[1], 10);
  const episode = parseInt(match[2], 10);
  const key = `${season},${episode}`;
  
  return EP_TITLE[key] || null;
}

/**
 * Format episode display string
 * @param {string} episodeId - Episode ID in format "S03E15"
 * @returns {string} Formatted string like "S3EP15 - Mr. Hankey's Christmas Classics"
 */
export function formatEpisodeDisplay(episodeId) {
  if (!episodeId) return episodeId;
  
  // Parse and format episode ID (e.g., "S03E15" -> "S3EP15")
  const match = episodeId.match(/S(\d+)E(\d+)/i);
  if (!match) return episodeId;
  
  const season = parseInt(match[1], 10);
  const episode = parseInt(match[2], 10);
  const formattedId = `S${season}EP${episode}`;
  
  const title = getEpisodeTitle(episodeId);
  if (title) {
    return `${formattedId} - ${title}`;
  }
  
  return formattedId;
}

export default EP_TITLE;

