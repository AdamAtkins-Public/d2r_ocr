import keyboard
import mouse
import time

def my_signal(key):
    print(key.__str__()+' was pressed')

def main_test():
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_UP and event.name == 'f1':
            my_signal('f1')
            print(mouse.get_position())
        elif event.event_type == keyboard.KEY_UP and event.name == 'f2':
            my_signal('f2')

            #screenshot of experience bar
            mouse.move(960,1000)
            time.sleep(0.5)
            keyboard.send('print screen')

        elif event.event_type == keyboard.KEY_UP and event.name == 'f7':
            my_signal('f7')
            break

if __name__ == '__main__':
    main_test()