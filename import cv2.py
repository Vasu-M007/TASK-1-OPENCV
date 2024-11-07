import cv2 as cv
import numpy as np
import sys
from math import atan2, cos, sin, sqrt, pi
 
img = cv.imread(r"C:\Users\vasum\OneDrive\Documents\y image.jpg",0)
ret,thresh = cv.threshold(img,127,255,0)
cnt1,_ = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img,cnt1,1,(55,255,255),4)
cv.imshow("Y",img)
cv.waitKey(0)

cap = cv.VideoCapture(r"C:\Users\vasum\Downloads\screen rec y4.mp4")
while True:
    _, frame = cap.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cnt2,_ = cv.findContours(gray_frame, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print("contours",len(cnt1))
    detection_found = False
    # cv.drawContours(frame, cnt2,-1,(255,0,0),3) 
    if cnt1 and cnt2:
    
        for c in cnt2:
            threshold = cv.matchShapes(cnt1[1],c,1,0.0)
            if threshold<0.3:
                cv.circle(frame,(60,55),3,(0,255,0),-1)
                cv.drawContours(frame,[c],0,(0,255,0),2) 
                        
                # detection_found = True
                rect=cv.minAreaRect(c)
                box=cv.boxPoints(rect)
                print("box points",box) 
                       
                box=np.array(box,dtype=np.uint64)
                         
                # box=np.int_(box)       
                cv.drawContours(frame, [box] , 0 , [255,0,0], 3)
                x1=box[0][0].item()
                y1=box[0][1].item()

                x2=box[1][0].item()
                y2=box[1][1].item()

                x3=box[2][0].item()
                y3=box[2][1].item()

                x4=box[3][0].item()
                y4=box[3][1].item()
                len1st = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                len2nd = np.sqrt((x3-x2)**2 + (y2-y1)**2)
                if len1st > len2nd:
                    axis_start = ((x1+x4)//2,(y1+y4)//2)
                    axis_end = ((x2+x3)//2,(y1+y4)//2)
                else:
                    axis_start = ((x1+x2)//2,(y1+y2)//2)
                    axis_end = ((x3+x4)//2,(y3+y4)//2)
                    
                cv.drawContours(frame, [box], 0, [255, 0, 0], 3)
                x, y, w, h = cv.boundingRect(c)
                vert_start = (x +(w//2), y)
                vert_end = (x +(w//2), y + h)
                
                dx = axis_end[0] - axis_start[0]
                dy = axis_end[1] - axis_start[1]
                line_angle = np.degrees(np.arctan2(dy, dx))   
                
                relative_angle = abs(90 - line_angle)
                if relative_angle > 90:
                    pass
                
                cv.line(frame, axis_start, axis_end, (255, 0, 0), 3)
                cv.line(frame, vert_start, vert_end, (0, 0, 255), 3) 
                
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.drawContours(frame, [c], -1, (255, 0, 0), 2)
                
                center_x = (x1 + x2 + x3 + x4) // 4
                center_y = (y1 + y2 + y3 + y4) // 4
                cv.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                
                cv.putText(frame, f"Angle: {relative_angle:.1f} deg",(x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX,0.9, (0, 255, 0), 2)
                
                break
            
    if detection_found:
        cv.circle(frame,(60,60),4,(0,255,0),-1)
    else:
        cv.circle(frame,(60,60),4,(0,0,255),-1)
        cv.imshow("resulted output",frame)
        k = cv.waitKey(27) 
        if k == ord("s"):
            break
cap.release()
cv.destroyAllwindows()
            
            

            



                

                
    



    
            
            
        
        
        
        

    


