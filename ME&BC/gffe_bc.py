import os
import numpy as np
import math
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data_io import read_exr, write_exr, read_matrix
from opengl_util import create_compute_shader, create_compute_program, create_texture, create_window, read_texture

def bind_background_buffer(texture_id, levels=4):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    for level in range(levels):
        glBindImageTexture(level, texture_id, level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

def gffe_bc(world_pos, world_pos_1, mv, color, depth, stencil, vp_matrix, vp_matrix_next,background_buffers, program):
    height, width = color.shape[0], color.shape[1]
    
    # 使用传入的ndarray中的数据创建纹理
    world_pos_tex = create_texture(world_pos, width, height)
    world_pos_1_tex = create_texture(world_pos_1, width, height)
    mv_tex = create_texture(mv, width, height)
    depth_tex = create_texture(depth, width, height)
    color_tex = create_texture(color, width, height)
    stencil_tex = create_texture(stencil, width, height)

    

    background_buffer_color0 = create_texture(background_buffers[0],width,height)
    background_buffer_color1 = create_texture(background_buffers[1],width//2,height//2)
    #background_buffer_tex2 = create_texture(background_buffers[2],width//4,height//4)
    #background_buffer_tex3 = create_texture(background_buffers[3],width//8,height//8)
    background_buffer_depth0 = create_texture(background_buffers[2],width,height)
    background_buffer_depth1 = create_texture(background_buffers[3],width//2,height//2)
    # 创建存放结果的纹理
    # warp_color_tex = create_texture(None, width, height)
    # warp_mv_tex = create_texture(None, width, height)
    # warp_depth_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)
    

    warp_background_buffer_color_tex0 = create_texture(None, width, height)
    warp_background_buffer_color_tex1 = create_texture(None, width//2, height//2)
    # warp_background_buffer_tex2 = create_texture(None, width//4, height//4)
    # warp_background_buffer_tex3 = create_texture(None, width//8, height//8)

    warp_background_buffer_depth_tex0 = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)
    warp_background_buffer_depth_tex1 = create_texture(None, width//2, height//2, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)

    # just for test
    # background_buffer_texture = glGenTextures(1)
    # bind_background_buffer(background_buffer_texture, levels=4)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    # glTexStorage2D(GL_TEXTURE_2D, 4, GL_RGBA32F, width, height)

    # 将纹理绑定到着色器

    
    in_textures = [world_pos_tex, world_pos_1_tex, mv_tex, depth_tex, color_tex, stencil_tex,background_buffer_color0,background_buffer_color1,background_buffer_depth0,background_buffer_depth1]
    glBindTextures(0, len(in_textures), in_textures)
    #out_textures = [warp_color_tex, warp_mv_tex, warp_depth_tex]
    out_textures = [warp_background_buffer_color_tex0,warp_background_buffer_color_tex1,warp_background_buffer_depth_tex0,warp_background_buffer_depth_tex1]
    glBindImageTextures(0, len(out_textures), out_textures)


    
    # 绑定uniform矩阵
    vp_loc = glGetUniformLocation(program, "vp_matrix")
    if vp_loc >= 0:
        glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix)
    vp_loc = glGetUniformLocation(program, "vp_matrix_next")
    if vp_loc >= 0:
        glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix_next)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)
    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # # 从结果纹理中读取数据
    # warp_color = read_texture(warp_color_tex, width, height)
    # warp_mv = read_texture(warp_mv_tex, width, height)
    # warp_depth = read_texture(warp_depth_tex, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT)
    # warp_depth = ((2147483647 - np.expand_dims(warp_depth, axis=-1)) / 65535).astype(np.float32)
    # warp_color = np.reshape(warp_color, (height, width, 4))
    # warp_mv = np.reshape(warp_mv, (height, width, 4))
    # warp_depth = np.reshape(warp_depth, (height, width, 1))
    warp_background_buffer_color_0 = read_texture(warp_background_buffer_color_tex0, width, height)
    warp_background_buffer_color_0 = np.reshape(warp_background_buffer_color_0, (height, width, 4))
    warp_background_buffer_color_1 = read_texture(warp_background_buffer_color_tex1, width//2, height//2)
    warp_background_buffer_color_1 = np.reshape(warp_background_buffer_color_1, (height//2, width//2, 4))

    warp_background_buffer_depth_0 = read_texture(warp_background_buffer_depth_tex0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT)
    warp_background_buffer_depth_0 = ((2147483647 - np.expand_dims(warp_background_buffer_depth_0, axis=-1)) / 65535).astype(np.float32)
    warp_background_buffer_depth_0 = np.reshape(warp_background_buffer_depth_0, (height, width, 1))
    warp_background_buffer_depth_1 = read_texture(warp_background_buffer_depth_tex1, width//2, height//2, GL_RED_INTEGER, GL_UNSIGNED_INT)
    warp_background_buffer_depth_1 = ((2147483647 - np.expand_dims(warp_background_buffer_depth_1, axis=-1)) / 65535).astype(np.float32)
    warp_background_buffer_depth_1 = np.reshape(warp_background_buffer_depth_1, (height//2, width//2, 1))
    # warp_background_buffer2 = read_texture(warp_background_buffer_tex2, width//4, height//4)
    # warp_background_buffer3 = read_texture(warp_background_buffer_tex3, width//8, height//8)


    glDeleteTextures(10, [world_pos_tex, world_pos_1_tex, mv_tex, depth_tex, color_tex, stencil_tex, warp_background_buffer_color_tex0,warp_background_buffer_color_tex1,warp_background_buffer_depth_tex0,warp_background_buffer_depth_tex1])
    glFinish()

    return warp_background_buffer_color_0,warp_background_buffer_color_1,warp_background_buffer_depth_0,warp_background_buffer_depth_1

def gffe_bc_init():
    # 初始化opengl和创建着色器
    create_window(1280, 720, "gffe_bc")
    with open("shader/gffe_bc.comp", "r") as f:
        shader_source = f.read()
    shader = create_compute_shader(shader_source)
    program = create_compute_program(shader)
    glUseProgram(program)
    return program


def save_background_buffer_to_exr_with_opencv(texture_id, width, height, filepath):
    # 绑定纹理
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # 创建一个空的 numpy 数组来存储纹理数据
    buffer = np.zeros((height, width, 4), dtype=np.float32)  # 假设 RGBA 格式

    # 从 GPU 中读取纹理数据
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buffer)

    # OpenCV expects data in HWC (Height, Width, Channels) format, and flips vertically
    buffer = np.flipud(buffer)  # OpenGL's origin is bottom-left, OpenCV uses top-left

    # OpenCV 保存为 EXR 格式 (RGBA)
    cv2.imwrite(filepath, buffer)
    print(f"Saved EXR file to {filepath}")

def create_empty_background_buffers(width, height, num_levels=2):
    """
    创建 4 张全黑的图片，每一层的分辨率递减。
    
    参数:
    - width: 第一层图片的宽度
    - height: 第一层图片的高度
    - num_levels: 总层数 (默认 4)
    
    返回:
    - buffers: 包含每层图片的列表
    """
    buffers = []
    current_width, current_height = width, height
    
    for _ in range(num_levels):
        # 创建全黑图片，格式为 RGBA (4 通道)
        buffer = np.zeros((current_height, current_width, 4), dtype=np.float32)
        buffers.append(buffer)
        
        # 下一层的分辨率减半
        current_width = max(1, current_width // 2)
        current_height = max(1, current_height // 2)
    current_width, current_height = width, height
    for _ in range(num_levels):
        # 创建全黑图片，格式为 RGBA (4 通道)
        buffer = np.zeros((current_height, current_width, 4), dtype=np.float32)
        buffers.append(buffer)
        
        # 下一层的分辨率减半
        current_width = max(1, current_width // 2)
        current_height = max(1, current_height // 2)
    return buffers

def gffe_bc_main(label_index, label_path, seq_path, save_path, scene_name, program, debug=False):
    """
    :param label_index: 需要预测的标签帧的索引
    :param label_path: 标签帧的路径
    :param seq_path: 序列帧的路径
    :param save_path: 保存路径
    :param scene_name: 场景名称
    :param program: 着色器程序
    """
    input_index = (label_index - 1) // 2
    world_pos = read_exr(os.path.join(label_path, f"{scene_name}WorldPosition.{str(label_index-1).zfill(4)}.exr"), channel=4)
    world_pos_1 = read_exr(os.path.join(label_path, f"{scene_name}WorldPosition.{str(label_index-3).zfill(4)}.exr"), channel=4)
    mv = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index).zfill(4)}.exr"), channel=4)
    depth = read_exr(os.path.join(label_path, f"{scene_name}SceneDepth.{str(label_index-1).zfill(4)}.exr"), channel=4)[:, :, 0:1]
    depth = np.repeat(depth, 4, axis=-1)
    color = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index-1).zfill(4)}.exr"), channel=4)
    stencil = read_exr(os.path.join(label_path, f"{scene_name}MyStencil.{str(label_index-1).zfill(4)}.exr"), channel=4)
    
    vp_matrix = read_matrix(os.path.join(label_path, f"{scene_name}Matrix.{str(label_index-1).zfill(4)}.txt"))
    vp_matrix_next = read_matrix(os.path.join(label_path, f"{scene_name}Matrix.{str(label_index).zfill(4)}.txt"))

    back_ground_buffers = create_empty_background_buffers(color.shape[1], color.shape[0])
    
    if(label_index-3 > 2):
        #back_ground_buffer = read_exr(os.path.join(save_path, f"{scene_name}BackgroundBuffer.{str(label_index-3).zfill(4)}.exr"))
        back_ground_buffers[0] = read_exr(os.path.join(save_path, f"{scene_name}BackgroundBufferColor0.{str(label_index-5).zfill(4)}.exr"), channel=4)
        back_ground_buffers[1] = read_exr(os.path.join(save_path, f"{scene_name}BackgroundBufferColor1.{str(label_index-5).zfill(4)}.exr"), channel=4)
        back_ground_buffers[2] = read_exr(os.path.join(save_path, f"{scene_name}BackgroundBufferDepth0.{str(label_index-5).zfill(4)}.exr"), channel=4)
        back_ground_buffers[3] = read_exr(os.path.join(save_path, f"{scene_name}BackgroundBufferDepth1.{str(label_index-5).zfill(4)}.exr"), channel=4)
    #warp_color, warp_mv, warp_depth = gffe_bc(world_pos, world_pos_1, mv, depth, color, stencil, vp_matrix, vp_matrix_next, program)
    b0,b1,b2,b3 = gffe_bc(world_pos, world_pos_1, mv, depth, color, stencil, vp_matrix, vp_matrix_next,back_ground_buffers, program)
    #print("debug flag")
    write_exr(os.path.join(save_path, f"{scene_name}BackgroundBufferColor0.{str(label_index-3).zfill(4)}.exr"),b0)
    write_exr(os.path.join(save_path, f"{scene_name}BackgroundBufferColor1.{str(label_index-3).zfill(4)}.exr"),b1)
    write_exr(os.path.join(save_path, f"{scene_name}BackgroundBufferDepth0.{str(label_index-3).zfill(4)}.exr"),b2)
    write_exr(os.path.join(save_path, f"{scene_name}BackgroundBufferDepth1.{str(label_index-3).zfill(4)}.exr"),b3)
    # write_exr(os.path.join(save_path, f"{scene_name}test.{str(label_index).zfill(4)}.exr"), test)
    # write_exr(os.path.join(save_path, f"{scene_name}WarpColor.{str(label_index).zfill(4)}.exr"), warp_color)
    # write_exr(os.path.join(save_path, f"{scene_name}WarpMotionVector.{str(label_index).zfill(4)}.exr"), warp_mv)

    # if debug:
    #     # 读取并保存label
    #     color_gt = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), channel=4)
    #     mv_gt = read_exr(os.path.join(label_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), channel=4)
    #     write_exr(os.path.join(save_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), color_gt)
    #     write_exr(os.path.join(save_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), mv_gt)