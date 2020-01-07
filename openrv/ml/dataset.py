class Dataset:

	def __init__(self, objects):
		for i in range(len(objects)):
			if type(objects[i]) != MlObject:
				objects[i] = MlObject(objects[i])

		self.dataset = list(objects)

	def shuffle(self):
		from random import shuffle
		self.dataset = shuffle(self.dataset)
		return self

	def split(self, test_size=0.5, random=False):
		if random:
			from random import shuffle
			shuffle(self.dataset)
		test_num = int(len(self.dataset) * test_size)
		train_num = int(len(self.dataset) * (1.-test_size))
		train, test = self[:train_num], self[test_num:]
		return train, test

	def set_labels(self, labels):
		for i, label in enumerate(labels):
			self.dataset[i].label = label

	@property
	def labels(self):
		labels = []
		for obj in self.dataset:
			labels.append(obj.label)
		return labels

	@property
	def datas(self):
		datas = []
		for obj in self.dataset:
			datas.append(obj.data)
		return datas

	def copy(self):
		return Dataset(self.dataset)

	def to_dict(self):
		from collections import defaultdict
		output = defaultdict(list)
		for obj in self.dataset:
			output[obj.label].append(obj.data)
		return dict(output)

	def to_list(self):
		output = []
		for obj in self.dataset:
			output.append((obj.label, obj.data))
		return output

	def __getitem__(self, val):
		if type(val) == int:
			return self.dataset[val]
		else:
			return Dataset(self.dataset[val])

	def __str__(self):
		return '<Dataset data: {}>'.format(', '.join(map(str, self.dataset)))

	def __repr__(self):
		return self.__str__()


class MlObject:

	def __init__(self, data, label=None):
		self.data = data
		self.label = label

	def __str__(self):
		return f'<MlObject label=\"{self.label}\" data={self.data}>'