from datetime import datetime
import time
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import*

model=YOLO('yolov8s.pt')
stream=cv2.VideoCapture('traffic.mp4')
# stream = CamGear(source='https://www.youtube.com/watch?v=nt3D26lrkho', stream_mode = True, logging=True).start() # YouTube Video URL as input
# stream = CamGear(source='https://www.youtube.com/watch?v=9bFOCNOarrA', stream_mode = True, logging=True).start() # YouTube Video URL as input
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

classnames = []
counter_down=[]
counter_up=[]
down={}
up={}

#đọc file coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
count_up=0
count_down=0
#đọc file tracker 
tracker =Tracker()
#khai báo vị trí khi đối tượng đi qua đó sẽ nhận diện 
area=[(497,240),(340,240),(70,360),(460,360)]
greenline=(288,267),(489,267)
#(489,267),(545,267)
greenline2=(532,323),(788,323)
area2=[(688,240),(532,240),(538,360),(857,360)]
area3=[(288,267),(489,267),(460,360),(70,360)]
area4=[(688,240),(532,240),(532,323),(788,323)]
while True:
    #đọc file vid dùng cái này    
    ret,frame = stream.read()
    if not ret:break  
    # #xài stream thì dùng cái này 
    # frame = stream.read()

    # tang toc do vid cai nay xài nó đi nhanh quÁ ko đọc kịp frame đâm ra nhận diện kém  
    # count += 1
    # if count % 3 != 0:
    #     continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
    #vòng lập lấy tọa độ lớp và tỉ lệ % trong 1 frame ảnh
        x1=int(row[0]) #toạ độ x trên
        y1=int(row[1]) #tọa độ y trên
        x2=int(row[2]) #tọa độ x dưới
        y2=int(row[3]) #tọa độ y dưới 
        p=float(row[4]) # này để lấy %
        d=int(row[5]) # lớp
        c=class_list[d]
        if 'car'  in c:
            list.append([x1,y1,x2,y2,p])
    #update list        
    bbox_id=tracker.update(list)
    # vòng lặp để lấy tâm của đối tượng
    for bbox in bbox_id:
        x3,y3,x4,y4,id,p=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        pp=p
         


        y=240
        z=360 
        offset = 7
        g=267
        h=323
        
        # mặc định là -1 nếu đối tượng đi qua vùng này thì nó sẽ thành 0 và thêm id nó vào biến down            
        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if result>=0:
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            cvzone.putTextRect(frame,f'{c, (round(pp*100,2))}',(x3,y3),1,1)

        if y < (cy + offset) and y > (cy - offset):
            down[id]=time.time()
            # print(down[id])
        if id in down:
                if z < (cy + offset) and z > (cy - offset)and pp>0.5:  
                    if counter_down.count(id)==0:
                        counter_down.append(id)
                if z < (cy + offset) and z > (cy - offset) and pp >0.5 and a_speed_kh>70: 
                    car_image = frame[y3:y4, x3:x4]
                        # Resize the cropped image to a larger size
                    resized_car_image = cv2.resize(car_image, (500, 500))  # Resize to 300x300 pixels or any desired size
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    cv2.imwrite(f"images/car_{timestamp}.jpg", resized_car_image)
                if g < (cy + offset) and g > (cy - offset)and pp>0.5:
                    distance=50 #meters
                    elapsed_time=time.time() - down[id]
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
        result1=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
        if result1>=0:
            cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) 



        # nếu đối tượng đi qua vùng này thì thêm id nó vào biến up                    
        result2=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)                
        if result2>=0:
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            cvzone.putTextRect(frame,f'{c,round(pp*100,2)}',(x3,y3),1,1)
        if z < (cy + offset) and z > (cy - offset):
            up[id]=time.time()
        if id in up:
           if y < (cy + offset) and y > (cy - offset) and pp >0.5:         
             if counter_up.count(id)==0:
                counter_up.append(id)            
                # print(counter_down)
           if y < (cy + offset) and y > (cy - offset) and pp >0.5 and a_speed_kh1>70: 
               car_image = frame[y3:y4, x3:x4]
                # Resize the cropped image to a larger size
               resized_car_image = cv2.resize(car_image, (500, 500))  # Resize to 300x300 pixels or any desired size
               timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
               cv2.imwrite(f"images/car_{timestamp}.jpg", resized_car_image)     
           if h < (cy + offset) and h > (cy - offset) and pp >0.5 :
               distance=20 #meters
               elapsed_time1=time.time() - up[id]
               a_speed_ms1 = distance / elapsed_time1
               a_speed_kh1 = a_speed_ms1 * 3.6
        result3=cv2.pointPolygonTest(np.array(area4,np.int32),((cx,cy)),False)
        if result3>=0:
            cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) 
    #màu mè hoa lá hẹ 
    text_color = (255,255,255)  # white color for text
    green_color = (0, 255, 0)  # (B, G, R)
    red_color = (0, 0, 255)  # (B, G, R)   
    blue_color = (255, 0, 0)  # (B, G, R) 

    cv2.line(frame,(340,y),(688,y),red_color,3)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,('detect'),(340,240),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame,(288,g),(489,g),green_color,3)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,('begin show speed'),(288,267),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame,(532,323),(788,323),green_color,3)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,('begin show speed'),(532,323),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame,(70,z),(857,z),blue_color,3)  # seconde line
    cv2.putText(frame,('end and count'),(70,360),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)  

    # 2 dòng này vẽ cái khung khi đối tượng đi dô để detect cho dễ nhìn
    # cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,255),2)
    # cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
    # cv2.polylines(frame,[np.array(area3,np.int32)],True,(255,255,255),2)
    # cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,0),2)

    #dùng hàm len() để đếm và vẽ lên màn hình 
    downwards = (len(counter_down))
    cv2.putText(frame,('going down - ')+ str(downwards),(60,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA)    
    # tương tự như trên
    downwardss = (len(counter_up))
    cv2.putText(frame,('going up - ')+ str(downwardss),(60,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA) 
    cv2.imshow("RGB", frame)


    if cv2.waitKey(1)&0xFF==27:
        break
stream.release()
cv2.destroyAllWindows()
