from setuptools import setup, find_packages

setup(
  name = 'linear_attention_transformer',
  packages = find_packages(exclude=['examples']),
  version = '0.9.0',
  license='MIT',
  description = 'Linear Attention Transformer',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/linear-attention-transformer',
  keywords = ['transformers', 'attention', 'artificial intelligence'],
  install_requires=[
      'torch',
      'axial-positional-embedding',
      'product-key-memory>=0.1.5'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)