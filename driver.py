#!/bin/python
import sys
from subprocess import Popen, PIPE
REPEAT_MEASURE_NB=10

# IF YOU ARE RUNNING ON THE DAS 5 set ON_DAS5 to True
# IF YOU ARE LOCALLY set ON_DAS5 to False
# Your code will be tested on the DAS5.


ON_DAS5=False

benchmark_list_das5=[{"id":"reference","filename":"prun -np 1 ./k_nearest","is_reference":True,"benchmark_results":[]},\
        {"id":"optimized sequential","filename":"prun -np 1 ./k_nearest_seq","is_reference":False,"benchmark_results":[]},\
        {"id":"optimized SIMD","filename":"prun -np 1 ./k_nearest_simd","is_reference":False,"benchmark_results":[]},\
        {"id":"optimized multi thread","filename":"prun -np 1 ./k_nearest_thread","is_reference":False,"benchmark_results":[]}]

benchmark_list_local=[{"id":"reference","filename":"./k_nearest","is_reference":True,"benchmark_results":[]},\
        {"id":"optimized sequential","filename":"./k_nearest_seq","is_reference":False,"benchmark_results":[]},\
        {"id":"optimized SIMD","filename":"./k_nearest_simd","is_reference":False,"benchmark_results":[]},\
        {"id":"optimized multi thread","filename":"./k_nearest_thread","is_reference":False,"benchmark_results":[]}]

def check_prun_loaded():
    ps=Popen(["prun 2>&1 "], shell=True ,stdout=PIPE)
    string = ps.stdout.readline()
    if string == b'/bin/sh: prun: command not found\n':
        return 0
    else:
        return 1

def benchmark_bin(filename,is_reference):
    sum_MD=0
    sum_ED=0
    sum_CS=0
    avg_MD=0
    avg_ED=0
    avg_CS=0
    min_MD=-1
    max_MD=-1
    min_ED=-1
    max_ED=-1
    min_CS=-1
    max_CS=-1
    if is_reference:
        match_string="reference "
    else:
        match_string="optimized "
    for i in range(0,REPEAT_MEASURE_NB):
        ps=Popen([filename], shell=True ,stdout=PIPE)
        for line in map(lambda x:str(x),ps.stdout.readlines()):
            if match_string+"MD" in line:
                stripped_line = line.replace('\\n','').replace("'","").replace(" ","")
                MD_time = float(stripped_line.split(":")[1])
                #print("Reference MD : "+str(MD_time))
                sum_MD+=MD_time
                if MD_time > max_MD:
                    max_MD=MD_time
                if MD_time < min_MD or min_MD == -1:
                    min_MD=MD_time
            if match_string+"ED" in line:
                stripped_line = line.replace('\\n','').replace("'","").replace(" ","")
                ED_time = float(stripped_line.split(":")[1])
                #print("Reference ED : "+str(ED_time))
                sum_ED+=ED_time
                if ED_time > max_ED:
                    max_ED=ED_time
                if ED_time < min_ED or min_ED == -1:
                    min_ED=ED_time
            if match_string+"CS" in line:
                stripped_line = line.replace('\\n','').replace("'","").replace(" ","")
                CS_time = float(stripped_line.split(":")[1])
                #print("Reference CS : "+str(CS_time))
                sum_CS+=CS_time
                if CS_time > max_CS:
                    max_CS=CS_time
                if CS_time < min_CS or min_CS == -1:
                    min_CS=CS_time
    if sum_MD==0:
        print("Problems benchmarking MD in "+filename)
    if sum_ED==0:
        print("Problems benchmarking ED in "+filename)
    if sum_CS==0:
        print("Problems benchmarking CS in "+filename)


    avg_MD=sum_MD/REPEAT_MEASURE_NB
    avg_ED=sum_ED/REPEAT_MEASURE_NB
    avg_CS=sum_CS/REPEAT_MEASURE_NB
    res={"min_MD":min_MD,"avg_MD":avg_MD,"max_MD":max_MD,"min_ED":min_ED,"avg_ED":avg_ED,"max_ED":max_ED,"min_CS":min_CS,"avg_CS":avg_CS,"max_CS":max_CS}
    return res


def print_table(benchmark_list):
    reference=""
    column_lenght={}
    total_lenght=0
    order=["id","min_MD","avg_MD","max_MD","min_ED","avg_ED","max_ED","min_CS","avg_CS","max_CS","Speedup MD","Speedup ED","Speedup CS"]

    for benchmark in benchmark_list:
        if benchmark["is_reference"]:
            reference=benchmark
            reference["benchmark_results"]["Speedup MD"]=1
            reference["benchmark_results"]["Speedup ED"]=1
            reference["benchmark_results"]["Speedup CS"]=1
        else:
            benchmark["benchmark_results"]["Speedup MD"]=reference["benchmark_results"]["avg_MD"]/benchmark["benchmark_results"]["avg_MD"]
            benchmark["benchmark_results"]["Speedup ED"]=reference["benchmark_results"]["avg_ED"]/benchmark["benchmark_results"]["avg_ED"]
            benchmark["benchmark_results"]["Speedup CS"]=reference["benchmark_results"]["avg_CS"]/benchmark["benchmark_results"]["avg_CS"]
        #print(str(benchmark["benchmark_results"]))
        for key,value in benchmark["benchmark_results"].items():
            if not key in column_lenght:
                value_str = "{0:.2f}".format(value)
                column_lenght[key]=len(value_str)
            else:
                value_str = "{0:.2f}".format(value)
                if column_lenght[key]<len(value_str):
                    column_lenght[key]=len(value_str)
        if not "id" in column_lenght:
            column_lenght["id"]=len(benchmark["id"])
        else:
            if column_lenght["id"]<len(benchmark["id"]):
                column_lenght["id"]=len(benchmark["id"])

    for key in order:
        if column_lenght[key] < len(key):
            column_lenght[key]=len(key)

    for key,value in column_lenght.items():
        #print(value)
        total_lenght+=value
    total_lenght+=len(column_lenght)*3
    line =""
    #print(str(column_lenght))
    #print(str(reference))
    line+="_"*total_lenght
    print(line)
    header=""
    for key in order:
        value=column_lenght[key]
        header+=key+" "*(value-len(key))
        header+=" | "
    print(header)
    print(line)
    row=""
    value = reference["id"]
    tot_len=column_lenght["id"]
    row+=str(value)+" "*(tot_len-len(str(value)))
    row+=" | "
    for key in order:
        if not key == "id":
            value = "{0:.2f}".format(reference["benchmark_results"][key])
            tot_len=column_lenght[key]
            row+=value+" "*(tot_len-len(value))
            row+=" | "
    print(row)
    print(line)
    for benchmark in benchmark_list:
        if benchmark == reference:
            continue
        row=""
        value = benchmark["id"]
        tot_len=column_lenght["id"]
        row+=str(value)+" "*(tot_len-len(str(value)))
        row+=" | "
        for key in order:
            if not key == "id":
                value = "{0:.2f}".format(benchmark["benchmark_results"][key])
                tot_len=column_lenght[key]
                row+=str(value)+" "*(tot_len-len(str(value)))
                row+=" | "
        print(row)
        print(line)


def dump_benchmark_info(benchmark,reference=None):
    if reference==None:
        print("Binary "+benchmark["id"])
        print("avg MD "+str(benchmark["benchmark_results"]["avg_MD"]))
        print("max MD "+str(benchmark["benchmark_results"]["max_MD"]))
        print("min MD "+str(benchmark["benchmark_results"]["min_MD"]))
        print("avg ED "+str(benchmark["benchmark_results"]["avg_ED"]))
        print("min ED "+str(benchmark["benchmark_results"]["min_ED"]))
        print("max ED "+str(benchmark["benchmark_results"]["max_ED"]))
        print("avg CS "+str(benchmark["benchmark_results"]["avg_CS"]))
        print("min CS "+str(benchmark["benchmark_results"]["min_CS"]))
        print("max CS "+str(benchmark["benchmark_results"]["max_CS"]))
    else:
        print("Binary "+benchmark["id"])
        print("avg MD "+str(benchmark["benchmark_results"]["avg_MD"]))
        print("max MD "+str(benchmark["benchmark_results"]["max_MD"]))
        print("min MD "+str(benchmark["benchmark_results"]["min_MD"]))
        print("speedup MD "+str(reference["benchmark_results"]["avg_MD"]/benchmark["benchmark_results"]["avg_MD"]))
        print("avg ED "+str(benchmark["benchmark_results"]["avg_ED"]))
        print("min ED "+str(benchmark["benchmark_results"]["min_ED"]))
        print("max ED "+str(benchmark["benchmark_results"]["max_ED"]))
        print("speedup ED "+str(reference["benchmark_results"]["avg_ED"]/benchmark["benchmark_results"]["avg_ED"]))
        print("avg CS "+str(benchmark["benchmark_results"]["avg_CS"]))
        print("min CS "+str(benchmark["benchmark_results"]["min_CS"]))
        print("max CS "+str(benchmark["benchmark_results"]["max_CS"]))
        print("speedup CS "+str(reference["benchmark_results"]["avg_CS"]/benchmark["benchmark_results"]["avg_CS"]))


if __name__=="__main__":
    if ON_DAS5:
        benchmark_list=benchmark_list_das5
        if not check_prun_loaded():
            print("prun module is not loaded\nPlease load:\nmodule load prun")
            sys.exit()
    else:
        benchmark_list=benchmark_list_local

    reference=""
    for benchmark in benchmark_list:
        print("benchmarking "+benchmark["id"]+" code")
        benchmark["benchmark_results"] = benchmark_bin(benchmark["filename"],benchmark["is_reference"])
        if benchmark["is_reference"]:
            reference=benchmark

    #dump_benchmark_info(reference)
    #for benchmark in benchmark_list:
    #    dump_benchmark_info(benchmark,reference=reference)

    print_table(benchmark_list)
