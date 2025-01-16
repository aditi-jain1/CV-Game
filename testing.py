import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pygame 

def main():
    pygame.init()
    cap = cv2.VideoCapture(0)
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    scale_factor = 0.3  # Adjust this to make camera feed smaller/larger
    display_cam_width = int(cam_width * scale_factor)
    display_cam_height = int(cam_height * scale_factor)
    cam_x = (WINDOW_WIDTH - display_cam_width) // 2
    cam_y = WINDOW_HEIGHT - display_cam_height - 20 
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hand Tracking")
    detector = HandDetector(maxHands=2, detectionCon=.8)
    BACKGROUND_COLOR = (40, 40, 40)
    LINE_COLOR = (0, 255, 0)
    LINE_THICKNESS = 2

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        screen.fill(BACKGROUND_COLOR)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        hands, fram = detector.findHands(frame)
        hand_points = None
        if hands:
            for hand in hands:
                lmList = hand["lmList"]
                bbox = hand["bbox"]
                center = hand["center"]
                handType = hand["type"]
                start_point = (int(lmList[7][0]), int(lmList[7][1]))  # Wrist
                end_point = (int(lmList[8][0]), int(lmList[8][1]))  
                hand_points = (
                    (int(start_point[0] * scale_factor) + cam_x, 
                     int(start_point[1] * scale_factor) + cam_y),
                    (int(end_point[0] * scale_factor) + cam_x, 
                     int(end_point[1] * scale_factor) + cam_y)
                )
                
                for lm in lmList:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 
                             5, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (int(lmList[7][0]), int(lmList[7][1])), 
                             5, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (int(lmList[8][0]), int(lmList[8][1])), 
                             5, (0, 255, 0), cv2.FILLED)
                    
        # Convert frame from BGR (OpenCV) to RGB (Pygame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to Pygame surface
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        
        # Scale down the camera feed
        frame_surface = pygame.transform.scale(frame_surface, (display_cam_width, display_cam_height))
        
        # Draw title text
        font = pygame.font.Font(None, 48)
        title = font.render("Hand Tracking Visualization", True, (255, 255, 255))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 50))
        screen.blit(title, title_rect)
        
        # Display the camera feed at the bottom middle
        screen.blit(frame_surface, (cam_x, cam_y))
        
        # Draw a border around the camera feed
        pygame.draw.rect(screen, (100, 100, 100), 
                        (cam_x-2, cam_y-2, display_cam_width+4, display_cam_height+4), 2)
        if hand_points:
            extended_start, extended_end = extend_line(hand_points[0], hand_points[1])
            pygame.draw.line(screen, LINE_COLOR, extended_start, extended_end, LINE_THICKNESS)
        pygame.display.flip()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def extend_line(p1, p2, scale=1000):
    diff_x = p2[0] - p1[0]
    diff_y = p2[1] - p1[1]
    extended_p1 = (
        int(p2[0] - scale * diff_x),
        int(p2[1] - scale * diff_y)
    )
    extended_p2 = (
        int(p2[0] + scale * diff_x),
        int(p2[1] + scale * diff_y)
    )
    return extended_p1, extended_p2

if __name__ == "__main__":
    main()