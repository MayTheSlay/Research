from evaluate import load

rouge = load("rouge")

# predictions = 
# references=

# results = rouge.compute(predictions=predictions,
#                          references=references,
#                             tokenizer=lambda x: x.split())
# print(results)