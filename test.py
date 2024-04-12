class A():
    def __init__(self) -> None:
        self.ab = 'hello'

class B(A):
    def __init__(self) -> None:
        super().__init__()
        self.bc = 'world'

A = A()
B = B()

print(A.ab)
print(B.ab)
print(B.bc)