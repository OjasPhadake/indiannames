# Provided by Ojas (project owner), 2026-09-06, as a third input to merge
# alongside the hand-curated marker lists in this directory and the
# Chaturvedi classifier's predictions -- see CITATIONS.md and
# src/expand_corpus.py for how this gets combined and re-ranked against
# real electoral-roll frequency data, not used as-is.
#
# UPDATE 2026-09-06: Ojas asked for a one-by-one QA pass against real
# electoral-roll data. 6 corrections applied below (each marked inline
# with what changed and why -- see DECISIONS.md #7 for the full audit,
# including cases checked and left alone because the evidence was
# ambiguous rather than a clear error). Everything else is still exactly
# as originally pasted.

FIRST_NAME_BANK = {
    "Hindu_Female": [
        "Priya", "Ananya", "Divya", "Pooja", "Shreya",
        "Neha", "Kavya", "Riya", "Aishwarya", "Sneha",
        "Deepa", "Meghna", "Sanya", "Tanvi", "Anjali",
        "Ishita", "Nandini", "Aditi", "Swati", "Ritika",
        "Vaishnavi", "Shruti", "Bhavna", "Madhuri", "Komal",
        "Pallavi", "Namrata", "Harini", "Lakshmi", "Soumya",
        "Isha", "Meera", "Radhika", "Kirti", "Rekha",
        "Sunita", "Geeta", "Manju", "Usha", "Archana",
        "Shalini", "Charu", "Jyoti", "Rupa", "Vandana",
    ],
    "Hindu_Male": [
        "Rahul", "Vikram", "Arjun", "Rohan", "Karthik",
        "Amit", "Suresh", "Rajesh", "Aditya", "Nikhil",
        "Varun", "Siddharth", "Manish", "Gaurav", "Deepak",
        "Akash", "Abhishek", "Ritesh", "Vivek", "Ankit",
        "Ashwin", "Pranav", "Harish", "Naveen", "Vishal",
        "Yash", "Raghav", "Chirag", "Tarun", "Shivam",
        "Ravi", "Anil", "Sunil", "Ramesh", "Mahesh",
        "Prakash", "Ajay", "Vijay", "Sandeep", "Rakesh",
        "Kunal", "Saurabh", "Neeraj", "Rohit", "Anand",
    ],
    "Muslim_Female": [
        "Fatima", "Zara", "Ayesha", "Nadia", "Shahida",  # was "Sana" -- real electoral-roll data is 72% male for Sana; Shahida is 99% female, classifier-confirmed Muslim (88%). See DECISIONS.md #7.
        "Rukhsar", "Afreen", "Hina", "Mariam", "Noor",
        "Saira", "Asma", "Bushra", "Shabana", "Rehana",
        "Alina", "Iqra", "Samina", "Humaira", "Meher",
        "Nilofer", "Rabia", "Farah", "Lubna", "Shazia",
        "Tasneem", "Yasmin", "Amina", "Naima", "Zainab",
        "Sabiha", "Nazia", "Ruksana", "Shaista", "Uzma",
        "Farida", "Kausar", "Sultana", "Gulnaz", "Rafia",
        "Ambreen", "Sameera", "Nargis", "Roshni", "Anisa",
    ],
    "Muslim_Male": [
        "Mohammed", "Imran", "Arif", "Danish", "Faisal",
        "Aamir", "Zubair", "Tariq", "Irfan", "Shahid",
        "Salman", "Rizwan", "Wasim", "Nadeem", "Junaid",
        "Faizan", "Bilal", "Sohail", "Sameer", "Yusuf",
        "Owais", "Hamza", "Adnan", "Azhar", "Nawaz",
        "Farhan", "Shadab", "Armaan", "Saif", "Aqib",
        "Kamran", "Rashid", "Anwar", "Zaid", "Ehsan",
        "Rafiq", "Naeem", "Rayyan", "Asif", "Kashif",
        "Mubashir", "Waqar", "Ismail", "Talha", "Zeeshan",
        "Tabrez", "Yasin", "Jamal", "Ilyas", "Mahmud",
        "Mansoor", "Sultan",
    ],
    # Sikh_Female/Sikh_Male: 2026-09-06 QA pass moved 9 names off Female
    # onto Male (Kuldeep, Manveer, Rajdeep, Ravneet, Rajinder, Ravinder,
    # Jasveer, Parminder, Navneet -- each is real-data majority-male,
    # 66-96% male, not the near-50/50 "genuinely unisex" pattern most of
    # this list actually is), moved Gurinder off Male onto Female (66%
    # female in real data), and removed Manpreet from Male (it was
    # already correctly listed under Female; real data is 70% female).
    # See DECISIONS.md #7.
    "Sikh_Female": [
        "Harpreet", "Simran", "Manpreet", "Gurpreet", "Jaspreet",
        "Sukhpreet", "Kirandeep", "Amandeep", "Jasleen", "Harsimran",
        "Satinder", "Navjot", "Ramandeep", "Harleen", "Gurleen",
        "Mandeep", "Prabhjot", "Amrit", "Baljeet", "Kanwal",
        "Charanjeet", "Supreet", "Ekjot", "Japneet", "Amanpreet",
        "Harnoor", "Simerjeet", "Manjot", "Karamjit", "Sukhjeet",
        "Rupinder", "Amanjot", "Jasneet", "Baljinder", "Simrat",
        "Ramanjit", "Gurinder",
    ],
    "Sikh_Male": [
        "Gurpreet", "Jaspreet", "Harjinder", "Kulwinder", "Sukhwinder",
        "Paramjit", "Balwinder", "Amarjit", "Navdeep", "Rajvir",
        "Tejinder", "Daljit", "Satnam", "Hardeep", "Inderjit",
        "Jagdeep", "Harmeet", "Mandeep", "Gurtej", "Arshdeep",
        "Dilpreet", "Prabhjot", "Kanwar", "Ranjit", "Gagandeep",
        "Charanjit", "Amritpal", "Jaswant", "Jagjit", "Balbir",
        "Sukhdev", "Amarinder", "Jaskaran", "Harvinder", "Ranjodh",
        "Simarpreet", "Gurmukh", "Baldev", "Rajpal", "Devinder",
        "Manjinder", "Sukhbir", "Jasbir", "Kuldeep", "Manveer",
        "Rajdeep", "Ravneet", "Rajinder", "Ravinder", "Jasveer",
        "Parminder", "Navneet",
    ],
    # Anita and Nisha removed 2026-09-06 -- both were also claimed by this
    # project's own Hindu marker list; Ojas's call was to keep them Hindu,
    # not Christian. See DECISIONS.md #9.
    "Christian_Female": [
        "Mary", "Flory", "Sonia", "Preethi",  # was "Rosario" -- real electoral-roll data is 93% male for Rosario; Flory is 99% female, classifier-confirmed Christian. See DECISIONS.md #7.
        "Lissy", "Sheena", "Bindu", "Teresa",
        "Clara", "Joanna", "Seema", "Shiny", "Cynthia",
        "Elizabeth", "Grace", "Jennifer", "Rebecca", "Rachel",
        "Sophia", "Angela", "Monica", "Gloria", "Helen",
        "Esther", "Ruth", "Ann", "Lydia", "Melanie",
        "Susan", "Tessy", "Alphonsa", "Rincy", "Merlin",
        "Reena", "Sherin", "Anila", "Beena", "Jessy",
        "Annie", "Celine", "Diana", "Irene", "Roseline",
    ],
    "Christian_Male": [
        "Joel", "Jijo", "Binu", "Shaji", "Joby",
        "Biju", "Tijo", "Sijo", "Renji", "Justin",  # was "Anoop" -- not distinctively Christian at all: 96% concentrated in North India with no Christian association in real data, and the classifier independently agrees (calls it Hindu, 47%). Justin is 99% male, classifier-confirmed Christian. See DECISIONS.md #7.
        "Kevin", "Ryan", "Samuel", "Daniel", "Joseph",
        "John", "Peter", "Thomas", "Mathew", "Andrew",
        "Aaron", "Nathan", "Neil", "Adrian", "Jerome",
        "Anthony", "Vincent", "Xavier", "Melvin", "Ivan",
        "Alwyn", "Errol", "Denzil", "Blaise", "Lawrence",
        "Terence", "Conrad", "Malcolm", "Gerald", "Wilfred",
        "Cajetan", "Savio", "Neville", "Trevor", "Clifford",
    ],
}

REGION_SURNAME_MAP = {
    # Hindu_North/West gained Bhatt as a second listing (Ojas's call:
    # Bhatt genuinely has real presence in both -- North per the data,
    # 66%; West per the traditional Gujarati-Brahmin association).
    # Hindu_West/South gained Shetty as a second listing likewise (West
    # per real data, 92%; South kept for the well-known Karnataka
    # association). See DECISIONS.md #7.
    "Hindu_North": [
        "Sharma", "Gupta", "Singh", "Yadav", "Tiwari", "Mishra",
        "Pandey", "Tripathi", "Srivastava", "Verma", "Agarwal",
        "Jaiswal", "Chauhan", "Saxena", "Bhardwaj", "Bhatt",
    ],
    "Hindu_South": [
        "Iyer", "Nair", "Pillai", "Rao", "Reddy",
        "Subramanian", "Narayanan", "Srinivasan", "Menon", "Gowda",
        "Acharya", "Hegde", "Shetty", "Krishnan", "Naidu",  # was "Prasad" -- not distinctively South at all: 70% concentrated in North India in real data (it's a generic pan-Indian Hindu name). Krishnan is genuinely Tamil/South-distinctive, classifier-confirmed Hindu (97%). See DECISIONS.md #7.
    ],
    "Hindu_West": [
        "Desai", "Patil", "Shah", "Mehta", "Joshi",
        "Kulkarni", "Pawar", "Chavan", "Jadhav", "Bhosale",
        "Vyas", "Trivedi", "Parikh", "Modi", "Bhatt",
        "Shetty",
    ],
    "Hindu_East": [
        "Banerjee", "Chatterjee", "Das", "Sen", "Roy",
        "Bose", "Mukherjee", "Bhattacharya", "Saha", "Ghosh",
        "Pal", "Mitra", "Paul", "Chakraborty", "Biswas",
        "Sikder",  # moved here from Muslim_Northeast -- Sikdar/Sikder is a real Hindu Bengali surname too, per Ojas's call. Note: real regional data actually shows this name 78% concentrated in South, not East/Bengal -- likely a spelling collision with an unrelated Southern name, kept as Hindu_East on cultural-knowledge grounds despite that. See DECISIONS.md #7.
    ],
    "Hindu_Northeast": [
        "Saikia", "Borah", "Gogoi", "Dutta", "Baruah",
        "Kalita", "Hazarika", "Mahanta", "Phukan", "Sarma",
        "Barman", "Deb", "Bora", "Bordoloi", "Chetia",
    ],
    # Muslim region lists: 2026-09-06, second QA pass (region-vs-real-data
    # check, see DECISIONS.md #7). Ansari and Jafri moved off North (real
    # data: Ansari is East/West-dominant, Jafri is West-dominant). Gazi,
    # Rangwala, Usmani moved onto North (all real-data North-dominant;
    # Rangwala had literally 0% real presence in West). Pasha moved onto
    # West (real-data West-dominant). Ahmed, Ansari, Chaudhary also added
    # to East per Ojas's "top-2 regions for huge names" call -- all three
    # are large enough to have real, substantial presence in more than
    # one region. Chaudhary removed from West (0% real presence there).
    "Muslim_North": [
        "Khan", "Siddiqui", "Qureshi", "Ahmed", "Hussain",
        "Mirza", "Syed", "Farooqui", "Hashmi", "Rehman",
        "Abbasi", "Alvi", "Azmi", "Kayani", "Lodhi",
        "Nadwi", "Rizvi", "Zaidi", "Bukhari", "Warsi",
        "Naqvi", "Kazmi", "Noorani", "Dehlvi", "Amrohi",
        "Chishti", "Ghori", "Iqbal", "Rangwala", "Usmani",
        "Gazi", "Bohra", "Chaudhary",
    ],
    # Muslim_South: Kutty removed 2026-09-06 -- also independently on this
    # project's own Christian marker list, and Ojas's call was to keep it
    # Christian (two-source backing there vs. one here). See DECISIONS.md #9.
    "Muslim_South": [
        "Rowther", "Marakkayar", "Lebbai", "Ravuthar", "Sait",
        "Kunhi", "Musaliar", "Thangal", "Haji", "Baig",
        "Moosa", "Koya", "Ismail", "Basheer",
        "Kunhali", "Marakkar", "Beary", "Bepari", "Yousuf",
        "Daniyal",
    ],
    # Muslim_West: Patel removed 2026-09-06 -- also independently on this
    # project's own Hindu marker list (same real West count on both
    # sides). Patel is overwhelmingly Hindu/Jain in reality; Ojas's call
    # was to keep it Hindu. See DECISIONS.md #9.
    "Muslim_West": [
        "Khoja", "Memon", "Vora", "Ghadiyali", "Attarwala",
        "Bagasrawala", "Kapadia", "Surtee", "Amreliwala", "Dholkawala",
        "Contractor", "Petiwala", "Rajkotwala", "Kazi",
        "Shaikh", "Jafri", "Pasha", "Ansari",
    ],
    "Muslim_East": [
        "Mondal", "Molla", "Sheikh", "Pramanik", "Munshi",
        "Sardar", "Talukdar", "Miah", "Mazumder", "Bhuiyan",
        "Sarkar", "Fakir", "Biswas", "Halder", "Ahmed",
        "Ansari", "Chaudhary",
    ],
    "Muslim_Northeast": [
        "Barbhuiya", "Laskar", "Borbhuiya", "Choudhury", "Uddin",
        "Bhuyan", "Rahman", "Ali", "Gani", "Islam",
        "Deka", "Nasir", "Karim", "Habib",
    ],
    "Sikh": [
        "Singh", "Kaur", "Gill", "Dhillon", "Sandhu", "Brar",
        "Sidhu", "Grewal", "Bedi", "Bajwa", "Cheema",
        "Chahal", "Sekhon", "Mann", "Pannu", "Randhawa",
        "Aulakh", "Sodhi", "Walia", "Khalsa",
        "Bhatti", "Toor", "Deol", "Dhaliwal", "Bal",
        "Dhanoa", "Gosal", "Hundal", "Johal", "Kang",
        "Khaira", "Klair", "Mahil", "Nijjar", "Rai",
        "Rehal", "Sahota", "Samra", "Sangha", "Sarai",
        "Sohal", "Uppal", "Virk", "Waraich", "Bains",
        "Bhogal", "Chana", "Dosanjh", "Gakhal", "Hayer",
        "Heer", "Jawanda", "Kahlon", "Kalirai", "Khangura",
        "Kooner", "Lidder", "Nagra", "Natt", "Pandher",
        "Purewal", "Sahni", "Sanghera", "Shergill", "Sudan",
        "Takhar", "Thandi", "Wahla", "Aujla", "Bhullar",
        "Chattha", "Dhadwal", "Garcha", "Ghuman", "Bassi",
    ],
    "Christian_South": [
        "Thomas", "Mathew", "Kurian", "Jose", "Abraham",
        "Varghese", "George", "John", "Philip", "Paul",
        "Joseph", "Chacko", "Jacob", "Koshy", "Cherian",
        "Daniel", "Stephen", "Antony", "Fernandez",
        "Ninan", "Zachariah", "Chandy", "Mammen", "Oommen",
        "Eapen", "Pothen", "Alexander", "Yohannan", "Skariah",
        "Mathan", "Kuriakose", "Thottathil", "Kutty", "Mani",
        "Samuel", "Punnoose", "Ittoop", "Wilson",  # was moved from Christian_North -- real data is 91% South, not North. See DECISIONS.md #7.
    ],
    "Christian_West": [
        "D'Souza", "Fernandes", "Rodrigues", "Lobo", "Pereira",
        "Mascarenhas", "Noronha", "Pinto", "Sequeira", "Monteiro",
        "Dias", "Barreto", "Gonsalves", "Menezes", "Colaco",
        "Costa", "Almeida", "D'Costa", "Furtado", "Rozario",
        "Vaz", "Braganza", "Fonseca", "Coelho", "Rebello",
        "Rasquinha", "Aguiar", "Carvalho", "Miranda", "Pais",
        "Gomes", "Lima", "Machado", "Saldanha", "Ribeiro",
        "Nazareth", "Athaide",
    ],
    # Christian_North: 2026-09-06, second QA pass. Removed Sahni (a
    # Punjabi Khatri Hindu/Sikh surname -- no real Christian association;
    # this project's own Sikh surname list already independently has
    # Sahni too, consistent with that). Removed Daniyal and Yousuf,
    # moved to Muslim_South -- both read as more Muslim/Urdu-associated
    # than Christian, and real data is 73-95% South, not North. Removed
    # Wilson, moved to Christian_South (91% real South). Added Isaac,
    # moved from Christian_South (75% real North, the opposite direction
    # of Wilson's move). See DECISIONS.md #7.
    # Bhatti and Gill removed 2026-09-06 -- both already independently on
    # this project's own Sikh surname list (Gill also on Ojas's own Sikh
    # list). Both are well-known Punjabi Jat clan names with no real
    # Christian association found; Ojas's call was to keep them Sikh.
    # See DECISIONS.md #9.
    "Christian_North": [
        "Masih", "Lal", "Prakash", "Chand",
        "Sagar", "Yaqub", "Yohanan", "Barkat", "Nazir",
        "Khokhar", "Anand", "Bhatia", "Chandra",
        "Elias", "Isaac",
    ],
    "Christian_East": [
        "Tirkey", "Toppo", "Kerketta", "Kujur", "Minj",
        "Lakra", "Ekka", "Barla", "Bara", "Xalxo",
        "Soreng", "Beck", "Dungdung", "Bakhla", "Horo",
        "Kindo", "Kandulna", "Purty", "Tigga", "Baxla",
    ],
    "Christian_Northeast": [
        "Ao", "Sema", "Angami", "Lotha", "Konyak",
        "Chang", "Zeliang", "Sangma", "Marak", "Momin",
        "Lyngdoh", "Nongrum", "Marbaniang", "Kharshiing", "Pariat",
        "Khongwir", "Suchiang", "Nongsiej", "Rongmei", "Kabui",
    ],
}
