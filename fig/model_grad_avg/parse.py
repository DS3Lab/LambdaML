file_name = "model_average"

f = open(file_name)

time_list = []
loss_list = []

for line in f:
    if line.__contains__("accuracy of the model"):
        ind1 = line.index("Time = ")
        ind2 = line.index(",", ind1)
        time = float(line[ind1+7:ind2])
        time_list.append(time)
        ind1 = line.index("loss = ")
        loss = float(line[ind1+7:])
        loss_list.append(loss)

print(time_list)
print(loss_list)
print("length = {}".format(len(time_list)))
