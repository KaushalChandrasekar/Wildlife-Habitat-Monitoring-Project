import os
import cv2
from ultralytics import YOLO

VIDEOS_DIR = 'C:/Users/kaush/OneDrive/Desktop'  

# Function to process the video
def process_video(video_name):
    video_path = os.path.join(VIDEOS_DIR, video_name)
    video_path_out = '{}_out.mp4'.format(video_path)

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model_path = os.path.join(VIDEOS_DIR, 'runs', 'detect', 'train15', 'weights', 'last.pt')

    # Load a model
    model = YOLO(model_path)  # load a custom model

    threshold = 0.5

    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {video_path_out}")

# Function to show menu and get user choice
def show_menu():
    print("Select the video to process:")
    print("1: Video 1")
    print("2: Video 2")
    print("3: Video 3")
    print("0: Exit")
    choice = input("Enter your choice: ")
    return choice

# Main function to drive the menu
def main():
    while True:
        choice = show_menu()
        if choice == '1':
            process_video('video1.mp4')
        elif choice == '2':
            process_video('video2.mp4')
        elif choice == '3':
            process_video('video3.mp4')
        elif choice == '4':
            process_video('video4.mp4')
        elif choice == '0':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
