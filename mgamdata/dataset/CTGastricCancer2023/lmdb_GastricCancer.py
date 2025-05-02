import logging
import os
import pickle
import pdb
from collections.abc import Callable
from io import BytesIO
from multiprocessing import Pool, cpu_count
from multiprocessing.managers import BaseManager
from tqdm import tqdm
from pprint import pprint
from lzma import compress, decompress, FORMAT_XZ
from typing import Any

import lmdb
import pydicom
import nrrd
import numpy as np

from mmengine.utils import ManagerMixin






class moving_average_filter(object):
	def __init__(self, window_size:int=100) -> None:
		self.window_size = window_size
		self.data = []
	
	def __call__(self, data:float) -> float:
		self.data.append(data)
		if len(self.data) > self.window_size:
			self.data.pop(0)
		return sum(self.data) / len(self.data)



class LMDB_DataBackend:
	COMPRESS_FORMAT = FORMAT_XZ				# 压缩格式
	COMPRESS_PRESET = 5						# 压缩等级
	MAX_MAP_SIZE = 1024 * 1024 * 1024 * 72	# LMDB容量：72G
	PIXEL_ARRAY_TYPE = np.int16				# 图像像素格式

	def __init__(self, dataset_root:str, lmdb_path:str, mode:str|None=None):
		self.dataset_root = dataset_root
		self.lmdb_path = lmdb_path
		print(f"正在加载LMDB数据库{lmdb_path}")
		# 检查该文件是否存在
		if mode=="init":
			self._init_lmdb()
		elif mode=="normal" or mode=='check':
			self.lmdb_env = lmdb.Environment(self.lmdb_path, LMDB_DataBackend.MAX_MAP_SIZE, 
											 readonly=True, create=False, lock=False, readahead=False)
		elif mode=="debug":
			self.lmdb_env = lmdb.Environment(self.lmdb_path, LMDB_DataBackend.MAX_MAP_SIZE, 
											 readonly=False, create=False)
		else:
			raise NotImplementedError("mode must be in ['init', 'normal']")
		print(f"LMDB数据库env已建立")

		if mode=='check':
			self._check_all_items()


	@classmethod
	def _check_one_item(cls, key, value) -> dict[str, Any] | None:
		try:
			if key.decode().startswith('METADATA'):
				decompressed_meta = cls._decompress_meta(value)
				if not isinstance(decompressed_meta, dict):
					return {'Error Type':'Invalid Decompressed Meta',
							'key': key.decode(),
							'decompressed': decompressed_meta}
			elif key.decode().startswith('PIXEL_ARRAY'):
				decompressed_pixel = cls._decompress_meta(value)
				if not isinstance(decompressed_pixel, dict):
					return {'Error Type':'Invalid Decompressed Pixel',
							'key': key.decode(),
							'decompressed': decompressed_pixel}
			else:
				return {'Error Type':'Invalid Key',
						'key': key.decode(),
						'decompressed': None}
			
			return None
			
		except Exception as e:
			return {'Error Type': e,
					'key': key.decode(),
					'decompressed': None}

	def _check_all_items(self):
		txn = self.lmdb_env.begin(write=False)
		cursor = txn.cursor()
		results = []
		failed_count = 0
		logger = logging.getLogger('lmdb_check')
		logging.basicConfig(filename='lmdb_check.log', filemode='w', level=logging.DEBUG)
		console_handler = logging.StreamHandler()
		logger.addHandler(console_handler)
		
		with Pool(cpu_count()) as p:
			pbar = tqdm(cursor, desc='部署任务')
			for key, value in pbar:
				results.append(p.apply_async(self._check_one_item, args=(key, value)))
			pbar = tqdm(results, desc='收集结果', total=len(results))
			for info in pbar:
				info = info.get()
				if info is not None: 
					pprint(info)
					logger.debug(info)
					failed_count += 1
			
		logger.info(f"{failed_count} failed")

	@classmethod
	def _init_lmdb_process_one_file(cls, params) -> dict[str|bytes, bytes|float]:
		dataset_root, file_root, file = params
		source_type = file.split(".")[-1]
		source_path = os.path.join(file_root, file)
		npy_path = source_path.replace(".dcm", ".npy").replace(".nrrd", ".npy")
		
		metadata_key_name, pixel_array_key_name = LMDB_DataBackend._map_path_key(dataset_root, source_path)

		# MetaData from dcm
		meta_buffer = BytesIO()
		if source_type == "dcm":
			dcm_data = pydicom.dcmread(source_path)
			meta = dcm_data.to_json_dict()
			meta.pop('7FE00010')	# 移除dcm中的像素数据，这部分已经由npy提供了
		elif source_type == "nrrd":
			pixel, meta = nrrd.read(source_path)
		else:
			raise NotImplementedError(f"source_type only support ['dcm', 'nrrd'], but got {source_type}")
		meta:dict
		pickle.dump(meta, meta_buffer)

		# Pixel from npy
		npy_buffer = BytesIO()
		npy_data = np.load(npy_path).astype(cls.PIXEL_ARRAY_TYPE)
		np.save(npy_buffer, npy_data, allow_pickle=False)

		# lzma压缩
		compressed_meta_byte = compress(meta_buffer.getvalue(), format=cls.COMPRESS_FORMAT, preset=cls.COMPRESS_PRESET)
		compressed_pixel_byte = compress(npy_buffer.getvalue(), format=cls.COMPRESS_FORMAT, preset=cls.COMPRESS_PRESET)
		compress_ratio = len(compressed_meta_byte+compressed_pixel_byte)/len(meta_buffer.getvalue()+npy_data.tobytes())

		return {metadata_key_name: compressed_meta_byte, 
				pixel_array_key_name: compressed_pixel_byte,
				"compress_ratio": compress_ratio}

	# 从零构建lmdb数据库
	def _init_lmdb(self):
		self.lmdb_env = lmdb.Environment(self.lmdb_path, LMDB_DataBackend.MAX_MAP_SIZE)
		compress_ratio_filter = moving_average_filter(window_size=100)

		from multiprocessing import Pool, cpu_count
		exec_pool = Pool(cpu_count())
		task_params = []
		txn = self.lmdb_env.begin(write=False)

		# 获取生成长度
		walk_step = 0
		for _ in os.walk(self.dataset_root):
			walk_step += 1
		pbar1 = tqdm(os.walk(self.dataset_root), desc="检查lmdb数据库完整性", total=walk_step)
		for roots, dirs, files in pbar1:
			for file in files:
				# 由于先前的数据处理中，数据集已经经过一次清洗和转换
				# 现在每一个dcm和nrrd文件都有一个对应名称的npy文件
				# 因此输出dcm和nrrd的名称，程序能够自动在目录下找到npy文件并读取进数据库
				if file.endswith(".dcm") or file.endswith(".nrrd"):	# 源序列读取
					name = self._map_path_key(self.dataset_root, os.path.join(roots, file))
					if not (txn.get(name[0], False) and txn.get(name[1], False)):
						task_params.append((self.dataset_root, roots, file))
		print(f"总共有{len(task_params)}个文件需要加入lmdb")
		
		# lmdb写入
		fetcher = exec_pool.imap_unordered(self._init_lmdb_process_one_file, task_params)
		pbar = tqdm(fetcher, total=len(task_params), desc="写入源序列")
		for bytes_dict in pbar:
			for key, value in bytes_dict.items():
				if key=="compress_ratio": 
					pbar.set_description(f"写入lmdb | 压缩率: {compress_ratio_filter(float(value))*100:.1f}%")
				else:
					with self.lmdb_env.begin(write=True) as txn:
						txn.put(key, value)
	
	@staticmethod
	def _map_path_key(dataset_root, path:str)->tuple[bytes, bytes]:
		path = path.split(os.path.basename(dataset_root))[-1]
		path = os.path.join(*path.replace("/","\\").split(".")[:-1])
		metadata_key_name = ("METADATA_" + path).encode()
		pixel_array_key_name = ("PIXEL_ARRAY_" + path).encode()
		return (metadata_key_name, pixel_array_key_name)


	def put(self, key:str, value:bytes):
		if not isinstance(value, bytes):
			raise TypeError("values put into lmdb must be bytes")
		
		compressed = compress(value, self.COMPRESS_FORMAT)
		with self.lmdb_env.begin(write=True) as txn:
			txn.put(key.encode(), compressed)
		
		print(f"WRITE TO LMDB database: {self.lmdb_env.path()}, key:{key}, ori_size: {len(value)/1024:.2f} KB, cmp_size: {len(compressed)/1024:.2f} KB")
		return True


	def get(self, key:str) -> bytes | None:
		with self.lmdb_env.begin(write=False) as txn:
			compressed = txn.get(key.encode(), None)
			if compressed is None:
				print(f"GET from LMDB database: {self.lmdb_env.path()}, key:{key}, KEY NOT FOUND!!! ")
				return None
			decompressed = decompress(compressed, self.COMPRESS_FORMAT) # type:ignore
			print(f"GET from LMDB database: {self.lmdb_env.path()}, key:{key}, ori_size: {len(decompressed)/1024:.2f} KB cmp_size: {len(compressed)/1024:.2f} KB")
			return decompressed


	def meta_data_dict(self, meta_path:str) -> dict:
		with self.lmdb_env.begin(write=False) as txn:
			meta_dict_key = 'REGISTRY_'+os.path.basename(meta_path)
			meta_buffer = txn.get(meta_dict_key.encode(), None)
			if meta_buffer is None:
				raise RuntimeError(f"GET from LMDB database: {self.lmdb_env.path()}, key:{meta_dict_key}, KEY NOT FOUND!!! ")
			else:
				return pickle.loads(decompress(meta_buffer, self.COMPRESS_FORMAT)) # type:ignore


	def database_test(self):
		# 打印lmdb数据库有关信息
		with self.lmdb_env.begin(write=False) as txn:
			print(f"LMDB数据库信息:{txn.stat()}") # type:ignore
			# 遍历数据库
			for key, value in txn.cursor():
				print(f"key:{key}")
				break
		
		meta, pixel = self(r"./img/柏家荣/ImageFileName000.dcm")	# type:ignore
		for key, value in meta.items():
			print(f"meta 键:{key} | 值:{meta.get(key)}")
		print(f"\npixel:{pixel}\n")


	@classmethod
	def _decompress_meta(cls, meta_buffer:bytes|None) -> dict | None:
		if meta_buffer is not None:
			meta_buffer = decompress(meta_buffer, format=cls.COMPRESS_FORMAT)
			return pickle.load(BytesIO(meta_buffer))
		else:
			return None

	@classmethod
	def _decompress_pixel(cls, pixel_buffer:bytes|None) -> np.ndarray | None:
		if pixel_buffer is not None:
			pixel_buffer = decompress(pixel_buffer, format=cls.COMPRESS_FORMAT)
			return np.load(BytesIO(pixel_buffer)).astype(cls.PIXEL_ARRAY_TYPE)
		else:
			return None

	@classmethod
	def decompress(cls, meta_buffer:bytes|None=None, pixel_buffer:bytes|None=None
                ) -> tuple[dict|None, np.ndarray|None]:
		meta = cls._decompress_meta(meta_buffer)
		pixel = cls._decompress_pixel(pixel_buffer)
		return (meta, pixel)


	def fetch_data(self, path:str, meta:bool=True, pixel:bool=True
                ) -> tuple[str, Any, Any]:
		meta_key, pixel_key = self._map_path_key(self.dataset_root, path)
		meta_buffer, pixel_buffer = None, None
		with self.lmdb_env.begin(write=False) as txn:
			if meta:
				meta_buffer = txn.get(meta_key, None)
			if pixel:
				pixel_buffer = txn.get(pixel_key, None)
		pdb.set_trace()
		return (path, meta_buffer, pixel_buffer)



class LMDB_DataBackend_MP_Manager(BaseManager):
	# 无需进行任何定义，但必须继承形成一个新的类
	pass


# 代理执行器
class LMDB_MP_Proxy(ManagerMixin):
	def __init__(self, name:str, lmdb_args:dict) -> None:
		super().__init__(name=name)

		self.lmdb_args = lmdb_args
		LMDB_DataBackend_MP_Manager.register('LMDB_DataBackend', LMDB_DataBackend)
		manager = LMDB_DataBackend_MP_Manager()
		manager.start()
		# 获取一个服务代理，这是一个共享对象
		self.lmdb_service = manager.LMDB_DataBackend(**lmdb_args)	# type:ignore


	def transform(self) -> Callable:
		raise NotImplementedError


	def __call__(self) -> LMDB_DataBackend:
		return self.lmdb_service
