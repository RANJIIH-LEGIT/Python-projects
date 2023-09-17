from tkinter import *
from tkinter import messagebox

root=Tk()
root.title('login')
root.geometry('925x500+300+200')
root.configure(bg="#fff")
root.resizable(False,False)
def sigin():
    username=user.get()
    password=code.get()

    if username=='admin' and password=='1234':
        screen=Toplevel(root)
        screen.title("Website")
        screen.geometry('925x500+300+200')
        screen.config(bg="white")
        Label(screen,text="HELLO EVERYONE!!",bg='#fff',font=('Calibri(body',50,'bold')).pack(expand=True)
        screen.mainloop()
    elif username!='admin' and password!=1234:
        messagebox.showerror("Invalid","invalid username and password")
    elif password!='1234':
        messagebox.showerror("Invalid","Invalid Password")
    elif username!='admin':
        messagebox.showerror("Invalid", "Invalid username")
img = PhotoImage(file='ranjith12.png')
Label(root,image=img,bg='white').place(x=100,y=100)
frame=Frame(root,width=400,height=450,bg='white')
frame.place(x=480,y=70)

heading=Label(frame,text='Sign in',fg='#57a1f8',bg='white',font=('Minion Web',22,'bold'))
heading.place(x=100,y=5)
################-------------------

def on_entry(e):
    user.delete(0, 'end')

def on_leave(e):
    name=user.get()
    if name=='':
        user.insert(0, 'Username')


user= Entry(frame,width=25,fg='black',border=0,bg="white",font=('Microsoft YaHei UI Light',11))
user.place(x=30,y=80)
user.insert(0,'Username')
user.bind('<FocusIn>', on_entry)
user.bind('<FocusOut>', on_leave)

Frame(frame,width=295,height=2,bg='black').place(x=25,y=107)
############---------------------
def on_entry(e):
    code.delete(0, 'end')

def on_leave(e):
    name=code.get()
    if name=='':
        code.insert(0, 'Password')
code= Entry(frame,width=25,fg='black',border=0,bg="white",font=('Microsoft YaHei UI Light',11))
code.place(x=30,y=150)
code.insert(0,'Password')
code.bind('<FocusIn>', on_entry)
code.bind('<FocusOut>', on_leave)

Frame(frame,width=295,height=2,bg='black').place(x=25,y=177)

##############-------------------
Button(frame,width=39,pady=7,text='Sign in',bg='#57a1f8',fg='white',border=0,command=sigin).place(x=35,y=204)
label=Label(frame,text="Don't have an account?",fg='black',bg='white',font=('Microsoft YaHei UI Light',9))
label.place(x=75,y=270)

sign_up=Button(frame,width=6,text='Sign up',border=0,bg='white',cursor='hand2',fg='#57a1f8')
sign_up.place(x=215,y=270)


root.mainloop()