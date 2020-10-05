import json

from tqdm import tqdm

def get_data(file_name, number):
    with open("data/test_med.txt", "rb") as f:
        input_data = f.read().decode("utf-8")
    f.close()
    input_data = input_data.split("\n\n")
    length = len(input_data)
    for rank in range(number):
        if rank == number - 1:
            data = input_data[int((rank / number) * length) : ]
        else:
            data = input_data[int((rank / number) * length) : int(((rank + 1) / number) * length)]
        f_out = open("data/test_med_multi/test_" + str(rank) + ".txt", "w")
        total = 0
        for utts in data:
            f_out.write(utts)
            total += 1
            if (total < len(data)):
                f_out.write("\n\n")
        f_out.close()
        print ("test_med_" + str(rank) + ".txt" + " finished.")
            
    

if __name__ == "__main__":
    get_data("test_med.txt", 12)
    print ("Finished.")
    
        
        


