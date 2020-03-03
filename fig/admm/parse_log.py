
def parse(file_name):
    f = open(file_name)
    time_list = []
    loss_list = []
    for line in f.readlines():
        if ", Time:" in line and ", Loss: " in line:
            ind_1 = line.index(", Time")
            ind_2 = line.index(", Loss", ind_1)
            cur_time = float(line[ind_1+8:ind_2])
            time_list.append(cur_time)
            ind_1 = line.index(", ", ind_2+1)
            cur_loss = float(line[ind_2+8:ind_1])
            loss_list.append(cur_loss)

    print(time_list)
    print(loss_list)


if __name__ == "__main__":
    parse("higgs_LR_admm_reduce_w_128_lr0.01_b1k.log")
    parse("higgs_LR_model_avg_w_128_lr0.01_b1k.log")
