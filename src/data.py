from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
BOOKS_PATH = DATA_DIR / "books.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"


@dataclass(frozen=True)
class BookSeed:
    title: str
    author: str
    genres: str
    moods: str
    pace: str
    length: str
    difficulty: str
    year: int
    description: str


BOOKS: list[BookSeed] = [
    BookSeed("Dune", "Frank Herbert", "science fiction;politics;adventure", "epic;strategic;immersive", "medium", "long", "medium", 1965, "Desert politics, ecology, prophecy, and power collide on a planet that controls the galaxy's most valuable resource."),
    BookSeed("The Left Hand of Darkness", "Ursula K. Le Guin", "science fiction;literary;anthropology", "thoughtful;wintry;philosophical", "slow", "medium", "high", 1969, "An envoy studies a world whose culture challenges fixed ideas of gender, loyalty, and belonging."),
    BookSeed("Project Hail Mary", "Andy Weir", "science fiction;adventure;humor", "clever;hopeful;fast", "fast", "medium", "medium", 2021, "A lone astronaut wakes with no memory and must solve an extinction-level problem with science and improvisation."),
    BookSeed("The Martian", "Andy Weir", "science fiction;survival;humor", "witty;resourceful;tense", "fast", "medium", "low", 2011, "A stranded astronaut uses engineering, botany, and stubborn optimism to survive Mars."),
    BookSeed("Neuromancer", "William Gibson", "science fiction;cyberpunk;noir", "gritty;stylish;dark", "fast", "medium", "high", 1984, "A washed-up hacker is pulled into a dangerous job involving artificial intelligence and corporate power."),
    BookSeed("Foundation", "Isaac Asimov", "science fiction;space opera;politics", "cerebral;sweeping;strategic", "medium", "medium", "medium", 1951, "Mathematicians try to preserve knowledge as a galactic empire moves toward collapse."),
    BookSeed("Hyperion", "Dan Simmons", "science fiction;literary;space opera", "mysterious;grand;haunting", "medium", "long", "high", 1989, "Pilgrims share stories while traveling toward a terrifying being on a distant world."),
    BookSeed("The Long Way to a Small Angry Planet", "Becky Chambers", "science fiction;cozy;found family", "warm;hopeful;character-driven", "slow", "medium", "low", 2014, "A spaceship crew forms tender bonds while tunneling pathways across the galaxy."),
    BookSeed("The Hobbit", "J.R.R. Tolkien", "fantasy;adventure;classic", "cozy;whimsical;brave", "medium", "medium", "low", 1937, "A comfort-loving hobbit joins dwarves and a wizard on a quest for treasure and courage."),
    BookSeed("The Fellowship of the Ring", "J.R.R. Tolkien", "fantasy;epic;adventure", "mythic;earnest;expansive", "slow", "long", "medium", 1954, "A fellowship carries a dangerous ring across a world shadowed by ancient evil."),
    BookSeed("A Game of Thrones", "George R.R. Martin", "fantasy;politics;epic", "dark;intricate;brutal", "medium", "long", "high", 1996, "Noble houses scheme for power while supernatural threats stir beyond the borderlands."),
    BookSeed("The Name of the Wind", "Patrick Rothfuss", "fantasy;coming of age;magic", "lyrical;romantic;intimate", "medium", "long", "medium", 2007, "A gifted musician and magician narrates the making and unmaking of his own legend."),
    BookSeed("Mistborn", "Brandon Sanderson", "fantasy;heist;magic", "inventive;fast;rebellious", "fast", "long", "medium", 2006, "A street thief joins a crew plotting to overthrow an immortal emperor through metal-based magic."),
    BookSeed("The Fifth Season", "N.K. Jemisin", "fantasy;dystopian;literary", "urgent;devastating;inventive", "medium", "long", "high", 2015, "In a world of recurring catastrophe, oppressed earth-shapers hold the key to survival."),
    BookSeed("Piranesi", "Susanna Clarke", "fantasy;mystery;literary", "dreamlike;lonely;wonder-filled", "slow", "short", "medium", 2020, "A gentle narrator explores an impossible house of endless halls, tides, and statues."),
    BookSeed("The Night Circus", "Erin Morgenstern", "fantasy;romance;historical", "lush;magical;romantic", "slow", "medium", "low", 2011, "Two young magicians compete within a mysterious traveling circus that opens only at night."),
    BookSeed("Pride and Prejudice", "Jane Austen", "classic;romance;social satire", "witty;bright;observant", "medium", "medium", "medium", 1813, "Misjudgments, manners, and attraction complicate the path to love and self-knowledge."),
    BookSeed("Jane Eyre", "Charlotte Bronte", "classic;gothic;romance", "brooding;resilient;intense", "slow", "long", "medium", 1847, "An orphaned governess seeks independence and love while guarding her moral center."),
    BookSeed("Wuthering Heights", "Emily Bronte", "classic;gothic;tragedy", "stormy;obsessive;dark", "slow", "medium", "high", 1847, "A fierce, destructive bond shapes generations on the Yorkshire moors."),
    BookSeed("Great Expectations", "Charles Dickens", "classic;coming of age;social", "melancholy;wry;reflective", "slow", "long", "medium", 1861, "An orphan's rise in fortune reveals shame, loyalty, and the limits of status."),
    BookSeed("The Great Gatsby", "F. Scott Fitzgerald", "classic;literary;tragedy", "glamorous;melancholy;sharp", "medium", "short", "medium", 1925, "Jazz Age wealth and longing gather around a mysterious millionaire and an impossible dream."),
    BookSeed("To Kill a Mockingbird", "Harper Lee", "classic;legal;coming of age", "compassionate;sober;humane", "medium", "medium", "low", 1960, "A child observes courage, prejudice, and justice in a small Southern town."),
    BookSeed("The Handmaid's Tale", "Margaret Atwood", "dystopian;literary;political", "chilling;controlled;urgent", "medium", "medium", "medium", 1985, "A woman narrates life under a theocratic regime that controls bodies and language."),
    BookSeed("Station Eleven", "Emily St. John Mandel", "literary;post-apocalyptic;speculative", "elegiac;hopeful;interwoven", "medium", "medium", "medium", 2014, "Artists and survivors carry memory, Shakespeare, and fragile civilization after a pandemic."),
    BookSeed("Cloud Atlas", "David Mitchell", "literary;speculative;historical", "ambitious;layered;playful", "medium", "long", "high", 2004, "Nested stories across centuries explore power, reincurrence, exploitation, and resistance."),
    BookSeed("The Road", "Cormac McCarthy", "literary;post-apocalyptic;survival", "bleak;tender;spare", "slow", "medium", "medium", 2006, "A father and son cross a ruined landscape carrying love through extreme scarcity."),
    BookSeed("Where the Crawdads Sing", "Delia Owens", "mystery;literary;nature", "atmospheric;lonely;suspenseful", "medium", "medium", "low", 2018, "A marsh-raised girl becomes entangled with love, isolation, and a murder investigation."),
    BookSeed("The Secret History", "Donna Tartt", "mystery;literary;dark academia", "elegant;sinister;intellectual", "slow", "long", "high", 1992, "A group of classics students hides a terrible act beneath beauty and privilege."),
    BookSeed("Gone Girl", "Gillian Flynn", "thriller;mystery;domestic noir", "twisty;cynical;sharp", "fast", "medium", "medium", 2012, "A marriage turns into a media spectacle after a woman's disappearance reveals layered deceptions."),
    BookSeed("The Girl with the Dragon Tattoo", "Stieg Larsson", "thriller;mystery;crime", "dark;investigative;propulsive", "medium", "long", "medium", 2005, "A journalist and hacker investigate a decades-old disappearance in a powerful family."),
    BookSeed("Big Little Lies", "Liane Moriarty", "mystery;domestic;social", "wry;suspenseful;sharp", "fast", "medium", "low", 2014, "Schoolyard politics and private secrets build toward a fatal trivia night."),
    BookSeed("The Silent Patient", "Alex Michaelides", "thriller;psychological;mystery", "tense;twisty;clinical", "fast", "medium", "low", 2019, "A psychotherapist tries to understand why a famous painter shot her husband and stopped speaking."),
    BookSeed("Rebecca", "Daphne du Maurier", "gothic;mystery;romance", "haunting;jealous;atmospheric", "slow", "medium", "medium", 1938, "A young bride enters a grand estate dominated by the memory of her husband's first wife."),
    BookSeed("The Thursday Murder Club", "Richard Osman", "mystery;cozy;humor", "warm;clever;playful", "fast", "medium", "low", 2020, "Retirees in a village community investigate murders with charm and tactical brilliance."),
    BookSeed("Educated", "Tara Westover", "memoir;education;family", "intense;reflective;resilient", "medium", "medium", "medium", 2018, "A woman raised in isolation pursues education and remakes her understanding of family."),
    BookSeed("Becoming", "Michelle Obama", "memoir;politics;inspiration", "warm;reflective;uplifting", "medium", "long", "low", 2018, "A former First Lady traces identity, public life, marriage, and purpose."),
    BookSeed("Born a Crime", "Trevor Noah", "memoir;humor;history", "funny;sharp;moving", "fast", "medium", "low", 2016, "Comedian Trevor Noah recounts childhood under apartheid and his mother's fierce love."),
    BookSeed("Sapiens", "Yuval Noah Harari", "nonfiction;history;big ideas", "broad;provocative;clear", "medium", "long", "medium", 2011, "A sweeping account of humanity's cognitive, agricultural, imperial, and scientific revolutions."),
    BookSeed("Atomic Habits", "James Clear", "self help;productivity;psychology", "practical;clear;motivating", "fast", "medium", "low", 2018, "Tiny behavior changes compound into stronger systems for identity, work, and life."),
    BookSeed("Thinking, Fast and Slow", "Daniel Kahneman", "psychology;nonfiction;behavioral economics", "analytical;dense;fascinating", "slow", "long", "high", 2011, "Research on cognitive biases explains how intuitive and deliberate thinking shape decisions."),
    BookSeed("Range", "David Epstein", "nonfiction;psychology;career", "curious;practical;expansive", "medium", "medium", "medium", 2019, "Generalists, sampling, and broad experience can outperform early specialization in complex fields."),
    BookSeed("The Power of Habit", "Charles Duhigg", "self help;psychology;business", "practical;narrative;clear", "medium", "medium", "low", 2012, "Stories and research explain how cues, routines, and rewards shape behavior."),
    BookSeed("Deep Work", "Cal Newport", "productivity;business;self help", "focused;practical;serious", "medium", "medium", "low", 2016, "A case for distraction-free concentration as a rare and valuable professional skill."),
    BookSeed("The Lean Startup", "Eric Ries", "business;startup;innovation", "practical;iterative;strategic", "medium", "medium", "low", 2011, "Entrepreneurs test assumptions quickly using experiments, feedback, and validated learning."),
    BookSeed("Zero to One", "Peter Thiel", "business;startup;strategy", "contrarian;concise;strategic", "medium", "short", "medium", 2014, "A startup manifesto about monopoly, technology, and creating new markets."),
    BookSeed("The Psychology of Money", "Morgan Housel", "finance;psychology;nonfiction", "wise;clear;grounded", "fast", "short", "low", 2020, "Short essays explore wealth, behavior, risk, luck, and long-term financial judgment."),
    BookSeed("A Brief History of Time", "Stephen Hawking", "science;physics;nonfiction", "cosmic;curious;challenging", "slow", "short", "high", 1988, "Black holes, time, cosmology, and the universe are explained for general readers."),
    BookSeed("The Immortal Life of Henrietta Lacks", "Rebecca Skloot", "science;biography;ethics", "human;investigative;moving", "medium", "medium", "medium", 2010, "The story of HeLa cells connects medical discovery, consent, race, and family history."),
    BookSeed("Braiding Sweetgrass", "Robin Wall Kimmerer", "nature;memoir;science", "gentle;wise;lyrical", "slow", "medium", "medium", 2013, "Botany, Indigenous knowledge, and gratitude become a way of seeing the living world."),
    BookSeed("Circe", "Madeline Miller", "mythology;fantasy;literary", "lyrical;feminist;lonely", "slow", "medium", "medium", 2018, "A misunderstood witch from Greek myth claims power, craft, exile, and love."),
    BookSeed("The Song of Achilles", "Madeline Miller", "mythology;romance;historical", "tender;tragic;lyrical", "medium", "medium", "low", 2011, "Patroclus and Achilles grow from boyhood companionship into mythic, doomed love."),
    BookSeed("The Book Thief", "Markus Zusak", "historical;literary;war", "tender;sad;hopeful", "medium", "long", "medium", 2005, "A girl in Nazi Germany steals books while Death narrates love and loss."),
    BookSeed("All the Light We Cannot See", "Anthony Doerr", "historical;literary;war", "delicate;moving;atmospheric", "medium", "long", "medium", 2014, "A blind French girl and German radio prodigy move through the devastation of World War II."),
    BookSeed("The Nightingale", "Kristin Hannah", "historical;war;family", "emotional;brave;dramatic", "medium", "long", "low", 2015, "Two sisters resist occupation in France through different kinds of courage."),
    BookSeed("Homegoing", "Yaa Gyasi", "historical;literary;family saga", "powerful;intergenerational;clear", "medium", "medium", "medium", 2016, "Two half-sisters' descendants trace the long effects of slavery across Ghana and America."),
    BookSeed("Pachinko", "Min Jin Lee", "historical;family saga;literary", "sweeping;tender;resilient", "medium", "long", "medium", 2017, "A Korean family in Japan survives exile, discrimination, ambition, and sacrifice over generations."),
    BookSeed("The Seven Husbands of Evelyn Hugo", "Taylor Jenkins Reid", "historical;romance;celebrity", "glamorous;emotional;confessional", "fast", "medium", "low", 2017, "An aging film star reveals the truth behind her marriages, ambition, and great love."),
    BookSeed("Tomorrow, and Tomorrow, and Tomorrow", "Gabrielle Zevin", "literary;friendship;gaming", "creative;bittersweet;intimate", "medium", "medium", "medium", 2022, "Two friends build video games and a complicated creative partnership over decades."),
    BookSeed("Normal People", "Sally Rooney", "literary;romance;coming of age", "intimate;melancholy;precise", "medium", "short", "medium", 2018, "Two young people move in and out of love, class tension, and self-understanding."),
    BookSeed("Lessons in Chemistry", "Bonnie Garmus", "historical;humor;feminist", "bright;witty;uplifting", "fast", "medium", "low", 2022, "A brilliant chemist becomes a television cooking host and challenges 1960s sexism."),
    BookSeed("Red, White & Royal Blue", "Casey McQuiston", "romance;comedy;politics", "sparkling;funny;hopeful", "fast", "medium", "low", 2019, "The son of a U.S. president falls for a British prince amid public pressure."),
    BookSeed("Beach Read", "Emily Henry", "romance;comedy;contemporary", "warm;witty;emotional", "fast", "medium", "low", 2020, "Two rival writers swap genres while confronting grief, chemistry, and creative blocks."),
    BookSeed("The Kiss Quotient", "Helen Hoang", "romance;contemporary;neurodiversity", "sweet;steamy;heartfelt", "fast", "medium", "low", 2018, "An econometrician hires an escort for dating lessons and finds unexpected intimacy."),
    BookSeed("The House in the Cerulean Sea", "TJ Klune", "fantasy;cozy;found family", "warm;whimsical;hopeful", "medium", "medium", "low", 2020, "A caseworker evaluates magical children and discovers a gentler version of courage."),
    BookSeed("Good Omens", "Terry Pratchett and Neil Gaiman", "fantasy;comedy;apocalypse", "absurd;clever;playful", "fast", "medium", "low", 1990, "An angel and demon try to prevent the apocalypse because they rather like Earth."),
    BookSeed("American Gods", "Neil Gaiman", "fantasy;mythology;road novel", "strange;dark;mythic", "medium", "long", "medium", 2001, "Old gods and new powers clash across an uncanny American landscape."),
    BookSeed("Neverwhere", "Neil Gaiman", "fantasy;urban;adventure", "dark;wonder-filled;quirky", "fast", "medium", "low", 1996, "A man slips into a hidden London of angels, monsters, and forgotten people."),
    BookSeed("Kindred", "Octavia E. Butler", "science fiction;historical;slavery", "urgent;harrowing;powerful", "fast", "medium", "medium", 1979, "A Black woman is repeatedly pulled through time to a plantation where survival has impossible costs."),
    BookSeed("Parable of the Sower", "Octavia E. Butler", "science fiction;dystopian;social", "urgent;visionary;bleak", "medium", "medium", "medium", 1993, "A young woman with hyperempathy builds a survival philosophy amid social collapse."),
    BookSeed("The Three-Body Problem", "Cixin Liu", "science fiction;hard sci-fi;first contact", "cerebral;vast;unsettling", "slow", "long", "high", 2008, "Humanity faces an alien civilization through physics, history, and strategic uncertainty."),
    BookSeed("Klara and the Sun", "Kazuo Ishiguro", "literary;science fiction;philosophical", "quiet;tender;strange", "slow", "medium", "medium", 2021, "An artificial friend observes love, illness, and what humans ask technology to become."),
    BookSeed("The Remains of the Day", "Kazuo Ishiguro", "literary;historical;character study", "restrained;melancholy;precise", "slow", "medium", "medium", 1989, "An English butler looks back on loyalty, missed love, and self-deception."),
    BookSeed("The Alchemist", "Paulo Coelho", "fable;inspiration;adventure", "simple;uplifting;spiritual", "fast", "short", "low", 1988, "A shepherd follows dreams, omens, and desire in search of treasure and purpose."),
    BookSeed("The Midnight Library", "Matt Haig", "speculative;inspiration;contemporary", "reflective;hopeful;accessible", "fast", "medium", "low", 2020, "A woman explores alternate lives in a library between life and death."),
    BookSeed("Anxious People", "Fredrik Backman", "contemporary;humor;ensemble", "warm;messy;compassionate", "fast", "medium", "low", 2019, "A failed bank robbery traps strangers whose vulnerabilities slowly connect."),
    BookSeed("A Man Called Ove", "Fredrik Backman", "contemporary;humor;community", "gruff;heartwarming;funny", "medium", "medium", "low", 2012, "A lonely curmudgeon is pulled back into life by neighbors who need him."),
    BookSeed("The Shadow of the Wind", "Carlos Ruiz Zafon", "historical;mystery;literary", "gothic;romantic;bookish", "medium", "long", "medium", 2001, "A boy discovers a forgotten novel and a dangerous mystery in postwar Barcelona."),
    BookSeed("The Priory of the Orange Tree", "Samantha Shannon", "fantasy;epic;dragons", "grand;queer;adventurous", "medium", "long", "medium", 2019, "Queens, mages, and dragon riders face a world-threatening ancient force."),
    BookSeed("Babel", "R.F. Kuang", "fantasy;historical;dark academia", "scholarly;angry;tragic", "slow", "long", "high", 2022, "Translation magic powers empire, forcing students to confront language, violence, and resistance."),
    BookSeed("Yellowface", "R.F. Kuang", "literary;satire;publishing", "sharp;uncomfortable;fast", "fast", "medium", "medium", 2023, "A writer steals a manuscript and spirals through ambition, racism, and literary scandal."),
    BookSeed("Sea of Tranquility", "Emily St. John Mandel", "speculative;literary;time travel", "elegiac;quiet;interwoven", "medium", "short", "medium", 2022, "Time, art, simulation, and pandemics echo through linked lives across centuries."),
]


def ensure_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(exist_ok=True)
    if not BOOKS_PATH.exists():
        books = pd.DataFrame([book.__dict__ for book in BOOKS])
        books.insert(0, "book_id", range(1, len(books) + 1))
        books.to_csv(BOOKS_PATH, index=False)
    else:
        books = pd.read_csv(BOOKS_PATH)

    if not RATINGS_PATH.exists():
        ratings = generate_interactions(books)
        ratings.to_csv(RATINGS_PATH, index=False)
    else:
        ratings = pd.read_csv(RATINGS_PATH)

    return books, ratings


def generate_interactions(books: pd.DataFrame, n_users: int = 900, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genre_vocab = sorted({g.strip() for genres in books["genres"] for g in genres.split(";")})
    mood_vocab = sorted({m.strip() for moods in books["moods"] for m in moods.split(";")})
    pace_values = sorted(books["pace"].unique())
    length_values = sorted(books["length"].unique())

    rows: list[dict[str, object]] = []
    for user_id in range(1, n_users + 1):
        favorite_genres = set(rng.choice(genre_vocab, size=rng.integers(3, 7), replace=False))
        favorite_moods = set(rng.choice(mood_vocab, size=rng.integers(2, 5), replace=False))
        pace_pref = rng.choice(pace_values)
        length_pref = rng.choice(length_values)
        adventurous = rng.beta(2, 4)

        scored_books = []
        for _, book in books.iterrows():
            genres = set(str(book["genres"]).split(";"))
            moods = set(str(book["moods"]).split(";"))
            genre_score = len(genres & favorite_genres) / max(len(genres), 1)
            mood_score = len(moods & favorite_moods) / max(len(moods), 1)
            pace_score = 0.18 if book["pace"] == pace_pref else 0
            length_score = 0.12 if book["length"] == length_pref else 0
            novelty = adventurous * rng.normal(0.15, 0.08)
            latent_score = 2.6 + 1.5 * genre_score + 0.9 * mood_score + pace_score + length_score + novelty + rng.normal(0, 0.35)
            scored_books.append((book["book_id"], float(np.clip(latent_score, 1, 5))))

        read_count = int(rng.integers(9, 28))
        probabilities = np.array([score for _, score in scored_books], dtype=float) ** 2.2
        probabilities = probabilities / probabilities.sum()
        chosen = rng.choice(len(scored_books), size=read_count, replace=False, p=probabilities)
        for index in chosen:
            book_id, score = scored_books[index]
            rows.append({"user_id": user_id, "book_id": int(book_id), "rating": round(score, 1)})

    return pd.DataFrame(rows)

