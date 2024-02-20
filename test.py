passage1 = [
    "This is passage 1",
    "this is the second part"
]
passage2 = [
    "this is passage 2",
    "This is passage 2 second part"
]

passage3 = [
    "this is passage 3",
    "This is the second part of passage 3"
]

test = [passage1, passage2, passage3]

zipped_passages = zip(passage1, passage2, passage3)
zipped_list = list(zipped_passages)

zip_test = zip(test)
zip_test_list = list(zip_test)

dictionary = {'example': 'give me one', 'example2': 'give me two'}

if 'dummy' in dictionary.keys():
    print("It is there!")
else:
    print("Its not there unfortunately!")

breakpoint()
