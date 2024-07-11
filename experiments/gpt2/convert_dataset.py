import json

output = []

with open("sms_spam_collection/SMSSpamCollection.tsv") as input_file:
    for line in input_file:
        label, text = line.split("\t")
        output.append({"text": text.strip(), "label": label})

with open("output.jsonl", "w") as output_file:
    for line in output:
        output_file.write(json.dumps(line) + "\n")
