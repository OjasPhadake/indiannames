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
    "Christian_Female": [
        "Mary", "Flory", "Anita", "Sonia", "Preethi",  # was "Rosario" -- real electoral-roll data is 93% male for Rosario; Flory is 99% female, classifier-confirmed Christian. See DECISIONS.md #7.
        "Lissy", "Sheena", "Nisha", "Bindu", "Teresa",
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
    "Hindu_North": [
        "Sharma", "Gupta", "Singh", "Yadav", "Tiwari", "Mishra",
        "Pandey", "Tripathi", "Srivastava", "Verma", "Agarwal",
        "Jaiswal", "Chauhan", "Saxena", "Bhardwaj",
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
    ],
    "Hindu_East": [
        "Banerjee", "Chatterjee", "Das", "Sen", "Roy",
        "Bose", "Mukherjee", "Bhattacharya", "Saha", "Ghosh",
        "Pal", "Mitra", "Paul", "Chakraborty", "Biswas",
    ],
    "Hindu_Northeast": [
        "Saikia", "Borah", "Gogoi", "Dutta", "Baruah",
        "Kalita", "Hazarika", "Mahanta", "Phukan", "Sarma",
        "Barman", "Deb", "Bora", "Bordoloi", "Chetia",
    ],
    "Muslim_North": [
        "Khan", "Ansari", "Siddiqui", "Qureshi", "Ahmed",
        "Hussain", "Mirza", "Syed", "Farooqui", "Hashmi",
        "Rehman", "Abbasi", "Alvi", "Azmi", "Jafri",
        "Kayani", "Lodhi", "Nadwi", "Rizvi", "Zaidi",
        "Bukhari", "Warsi", "Naqvi", "Kazmi", "Noorani",
        "Dehlvi", "Amrohi", "Chishti", "Ghori", "Iqbal",
    ],
    "Muslim_South": [
        "Rowther", "Marakkayar", "Lebbai", "Ravuthar", "Sait",
        "Kunhi", "Musaliar", "Thangal", "Haji", "Pasha",
        "Baig", "Moosa", "Kutty", "Koya", "Ismail",
        "Basheer", "Kunhali", "Marakkar", "Beary", "Bepari",
    ],
    "Muslim_West": [
        "Bohra", "Khoja", "Memon", "Vora", "Rangwala",
        "Ghadiyali", "Attarwala", "Bagasrawala", "Kapadia", "Surtee",
        "Amreliwala", "Dholkawala", "Contractor", "Petiwala", "Rajkotwala",
        "Chaudhary", "Kazi", "Usmani", "Patel", "Shaikh",
    ],
    "Muslim_East": [
        "Mondal", "Molla", "Sheikh", "Pramanik", "Munshi",
        "Sardar", "Talukdar", "Miah", "Mazumder", "Bhuiyan",
        "Sarkar", "Fakir", "Gazi", "Biswas", "Halder",
    ],
    "Muslim_Northeast": [
        "Barbhuiya", "Laskar", "Borbhuiya", "Choudhury", "Uddin",
        "Bhuyan", "Rahman", "Ali", "Gani", "Islam",
        "Sikder", "Deka", "Nasir", "Karim", "Habib",
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
        "Daniel", "Isaac", "Stephen", "Antony", "Fernandez",
        "Ninan", "Zachariah", "Chandy", "Mammen", "Oommen",
        "Eapen", "Pothen", "Alexander", "Yohannan", "Skariah",
        "Mathan", "Kuriakose", "Thottathil", "Kutty", "Mani",
        "Samuel", "Punnoose", "Ittoop",
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
    "Christian_North": [
        "Masih", "Lal", "Prakash", "Chand", "Bhatti",
        "Sagar", "Yaqub", "Yohanan", "Barkat", "Nazir",
        "Khokhar", "Gill", "Anand", "Daniyal", "Sahni",
        "Bhatia", "Wilson", "Chandra", "Elias", "Yousuf",
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
