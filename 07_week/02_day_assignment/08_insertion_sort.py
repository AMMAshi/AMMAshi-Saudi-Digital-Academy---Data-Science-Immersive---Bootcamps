#============================================
# Arwa Ashi - HW 2 - Week 7 - Oct 19, 2020
#============================================

# =========================================================
# Objectives: Execute insertion sort
# =========================================================
arr    = [50,32,2,77,25]
arr_01 = [500,2,300,99,10,8]

def insertion_sort(arr):
    for i in range(len(arr)):
        #print(i,arr[i])
        for j in range(1,len(arr)):
            if i < j:
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
                    #print("first if",i,j,arr)
                    if j == len(arr):
                        k= len(arr)
                        for i in  range(1,k,1):
                            if arr[k] < arr[k-i]:
                                arr[k], arr[k-i] = arr[k-i],arr[k]
                            #print("second if",i,j,k,arr)
            
    return arr

print(insertion_sort(arr))
print(insertion_sort(arr_01))




























