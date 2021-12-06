if __name__ == '__main__':
    file = open("C:/Users/User/Documents/Audacity/sample-data.txt", "r")
    redlined = (file.readlines())
    x = float(input("start time: "))
    y = float(input("end time: "))
    stepup = (y - x) / len(redlined)
    s = ""
    for i in range(len(redlined)):
        s += str(i * stepup) + "\t" + str(redlined[i])

    file.close()

    file = open(r"C:\Users\User\Downloads\waveform.txt", "w")
    #print(s)
    file.write(s)
    file.close()
    print("done")
