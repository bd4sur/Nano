const fs = require("fs");

function pinyin_to_keys(py) {
    const alphabet_to_key = {
        "a": 2, "b": 2, "c": 2,
        "d": 3, "e": 3, "f": 3,
        "g": 4, "h": 4, "i": 4,
        "j": 5, "k": 5, "l": 5,
        "m": 6, "n": 6, "o": 6,
        "p": 7, "q": 7, "r": 7, "s": 7,
        "t": 8, "u": 8, "v": 8,
        "w": 9, "x": 9, "y": 9, "z": 9,
    };
    let keys = "";
    for(let i = 0; i < py.length; i++) {
        keys += String(alphabet_to_key[py[i]]);
    }
    return keys;
}

let csv = fs.readFileSync("hanzi.csv", {"flag": "r", "encoding": "utf-8"});

let lines = csv.split("\n");

let UTF32_LIST = [];
let KEYS_LIST = [];

for(let line of lines) {
    let new_line = "";
    let fields = line.split("\t");
    let hanzi = fields[1].codePointAt(0);
    let pinyins = [...new Set(
        fields[4].split("/").map((v)=>v.toLowerCase()).map((v)=>v.replace(/[0-9]$/gi, ""))
    )];
    for(let i = 0; i < pinyins.length; i++) {
        let py = pinyins[i];
        let keys = Number(pinyin_to_keys(py));
        UTF32_LIST.push(hanzi);
        KEYS_LIST.push(keys);
    }
}

console.log(UTF32_LIST.length);

/*
fs.writeFileSync("list.txt", JSON.stringify({
    "UTF32_LIST": UTF32_LIST,
    "KEYS_LIST": KEYS_LIST
}, 2), {"encoding": "utf-8"})

*/