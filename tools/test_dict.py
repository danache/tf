from tools.keypoint_eval import load_annotations
a = [b'ff945ae2e729f24eea992814639d59b3bdec8bd8', b'97d4691271e6db6487fbab34ea28cec869833e2d', b'3243f9fbbd8de5f95a3f7feffc283f23f4be9352', b'6661fef2df4b03478fcc59885ebed82956e3ae6b', b'054d9ce9201beffc76e5ff2169d2af2f027002ca', b'd8eeddddcc042544a2570d4c452778b912726720', b'419b0b18b7f69b111b15495c632f5d2f4c8fc140', b'e7f7db1d9da0de281fc2933ba82f6a9f66640457', b'7f395cace9c9884ad35104c2933114928b33c71d', b'14b28b5725f543c4cbdcd32202a661630bc5c17e', b'b46b6003697d6f2876d7d7a1d553781e5d13548e', b'2067b837138c5ae79e02729ffe64a33ec08cbded', b'24c9af471d790f45f9c9bcfaa32c71418589d3cf', b'fa436c914fe4a8ec1ec5474af4d3820b84d17561', b'ced3f5370f950800d5171d81db3f412ab45bf276', b'cfce54a5c26506d02f2a1a9ba82d30af4122f641', b'37ebb4c1df8ad9c71d903679ef90d28a2b6b28f5', b'ce413cf87891f51d5ea356103b2d6f520f00f967', b'06245942cb3308359d1bd9867c83c3ac4af5bf3d', b'7ae79f65e9443b8d8499239e4a793389cfa73dd7', b'634be26aab321d2a3efc3edbe8020525f51e5a1b', b'8bdc278f579009950c32a6ab4a5b37b1fe9e6919']
return_dict = dict()
return_dict['error'] = None
return_dict['warning'] = []
return_dict['score'] = None
anno = load_annotations("/media/bnrc2/_backup/ai/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json"
, return_dict)


for item in a:
    item = item.decode("utf-8")
    if item in list(set(anno['image_ids'])):
        print ("in")