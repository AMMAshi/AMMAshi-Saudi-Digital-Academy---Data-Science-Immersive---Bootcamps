#============================================
# Arwa Ashi - HW 2 - Week 7 - Oct 19, 2020
#============================================

# ====================================================
# Objectives: Execute selection sort
# ====================================================

arr_01 = [8,5,2,9] # GOAL [2,5,8,9]
arr_02 = [9,3,5,6,90,2,7,10,99,987]
arr_03 = [999,300,500,2,700,10,1100]
arr_04 = [9,2,4,5,22,33,8]
arr_05 = [50,32,2,77,25]

def selection_sort(arr):
    for i in range(0,len(arr),1):
        hi = i
        for j in range(1,len(arr)):
            if j > i:
                if arr[j] < arr[hi]:
                    hi = j
                    #print(i,hi)
        arr[i],arr[hi] = arr[hi],arr[i]
        #print(arr)
        
            
    return arr

print(selection_sort(arr_01))
print(selection_sort(arr_02))
print(selection_sort(arr_03))
print(selection_sort(arr_04))
print(selection_sort(arr_05))





    
