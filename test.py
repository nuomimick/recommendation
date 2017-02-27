from recommend import datasets

class A:
	@property
	def say(self):
		return 'hello'

	def test(self):
		print(self.say)

a = A()
print(a.test())



