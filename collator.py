import sys

assert len(sys.argv) == 2, "Need to pass directory as command line argument"

answer = -1
with open(sys.argv[1] + f'/eval_results.txt', "r") as f:
    for line in f:
        strs = line.split()
        if strs[0] == "masked_lm_loss":
            answer = strs[2]

with open("collated_val.txt", "a+") as f:
    f.write(f'{answer}\n')
