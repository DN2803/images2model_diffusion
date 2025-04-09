# Description: Gradio app for 3D object reconstruction and pose estimation from sparse views.
import os
import sys
import json
import uuid
import numpy as np
import gradio as gr
import trimesh
import zipfile
import subprocess
from datetime import datetime
from functools import partial
from PIL import Image, ImageChops
import open3d as o3d
# from huggingface_hub import snapshot_download

from utils.image import ImageUtils

from modules.pcl_generator.depth_image import DepthImages
from modules.pcl_generator.main import PCL

# from gradio_model3dcolor import Model3DColor
# from gradio_model3dnormal import Model3DNormal

is_local_run = True
if is_local_run:
    code_dir = os.path.dirname(__file__)

if not is_local_run:
    zip_file_path = f"{code_dir}/examples.zip"
    # Unzipping the file into the current directory
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(os.getcwd())


_TITLE = (
    """Fast 3D Object Reconstruction and Pose Estimation from Sparse Views"""
)
_DESCRIPTION = (
    """Reconstruct 3D textured mesh from one or a few unposed images!"""
)



STYLE = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    <style>
        .alert, .alert div, .alert b {
            color: black !important;
        }
    </style>
"""
# info (info-circle-fill), cursor (hand-index-thumb), wait (hourglass-split), done (check-circle)
ICONS = {
    "info": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#0d6efd" class="bi bi-info-circle-fill flex-shrink-0 me-2" viewBox="0 0 16 16">
    <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>
    </svg>""",
    "cursor": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#0dcaf0" class="bi bi-hand-index-thumb-fill flex-shrink-0 me-2" viewBox="0 0 16 16">
    <path d="M8.5 1.75v2.716l.047-.002c.312-.012.742-.016 1.051.046.28.056.543.18.738.288.273.152.456.385.56.642l.132-.012c.312-.024.794-.038 1.158.108.37.148.689.487.88.716.075.09.141.175.195.248h.582a2 2 0 0 1 1.99 2.199l-.272 2.715a3.5 3.5 0 0 1-.444 1.389l-1.395 2.441A1.5 1.5 0 0 1 12.42 16H6.118a1.5 1.5 0 0 1-1.342-.83l-1.215-2.43L1.07 8.589a1.517 1.517 0 0 1 2.373-1.852L5 8.293V1.75a1.75 1.75 0 0 1 3.5 0z"/>
    </svg>""",
    "wait": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#6c757d" class="bi bi-hourglass-split flex-shrink-0 me-2" viewBox="0 0 16 16">
    <path d="M2.5 15a.5.5 0 1 1 0-1h1v-1a4.5 4.5 0 0 1 2.557-4.06c.29-.139.443-.377.443-.59v-.7c0-.213-.154-.451-.443-.59A4.5 4.5 0 0 1 3.5 3V2h-1a.5.5 0 0 1 0-1h11a.5.5 0 0 1 0 1h-1v1a4.5 4.5 0 0 1-2.557 4.06c-.29.139-.443.377-.443.59v.7c0 .213.154.451.443.59A4.5 4.5 0 0 1 12.5 13v1h1a.5.5 0 0 1 0 1h-11zm2-13v1c0 .537.12 1.045.337 1.5h6.326c.216-.455.337-.963.337-1.5V2h-7zm3 6.35c0 .701-.478 1.236-1.011 1.492A3.5 3.5 0 0 0 4.5 13s.866-1.299 3-1.48V8.35zm1 0v3.17c2.134.181 3 1.48 3 1.48a3.5 3.5 0 0 0-1.989-3.158C8.978 9.586 8.5 9.052 8.5 8.351z"/>
    </svg>""",
    "done": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#198754" class="bi bi-check-circle-fill flex-shrink-0 me-2" viewBox="0 0 16 16">
    <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z"/>
    </svg>""",
}

icons2alert = {
    "info": "primary",  # blue
    "cursor": "info",  # light blue
    "wait": "secondary",  # gray
    "done": "success",  # green
}


def message(text, icon_type="info"):
    return f"""{STYLE}  <div class="alert alert-{icons2alert[icon_type]} d-flex align-items-center" role="alert"> {ICONS[icon_type]}
                            <div> 
                                {text} 
                            </div>
                        </div>"""


def create_tmp_dir():
    tmp_dir = (
        "../demo_exp/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + str(uuid.uuid4())[:4]
    )
    os.makedirs(tmp_dir, exist_ok=True)
    print("create tmp_exp_dir", tmp_dir)
    return tmp_dir

def center_crop_and_resize(img, target_size=(320, 320)):
    # Lấy kích thước ảnh gốc
    width, height = img.size
    
    # Tính toán vùng crop để lấy phần giữa của ảnh
    if width > height:
        left = (width - height) // 2
        top = 0
        right = left + height
        bottom = height
    else:
        left = 0
        top = (height - width) // 2
        right = width
        bottom = top + width

    # Cắt phần trung tâm của ảnh
    img = img.crop((left, top, right, bottom))

    # Resize ảnh về kích thước mong muốn
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    return img
def preprocess_imgs(tmp_dir, input_img):
    used_seg_dir = os.path.join(tmp_dir, "used_seg")
    os.makedirs(used_seg_dir, exist_ok=True)
    for i, img_tuple in enumerate(input_img):
        img = Image.open(img_tuple[0])
        img = center_crop_and_resize(img)
        img.save(f"{tmp_dir}/input_{i}.png")
        #TODO call segmentation API
        seg_img = ImageUtils.segment_img(img)
        seg_img.save(os.path.join(used_seg_dir, f"seg_{i}.png"))
    return [Image.open(os.path.join(used_seg_dir, f"seg_{i}.png")) for i in range(len(input_img))]

def ply_to_glb(ply_path):
    glb_path = ply_path.replace(".ply", ".glb")
    mesh = trimesh.load(ply_path)
    mesh.export(glb_path)
    return glb_path

def pcd_gen(tmp_dir, use_seg):
    #TODO: call API to generate point cloud
    seg_img_paths = []
    if use_seg:
        seg_img_paths = [f"{tmp_dir}/used_seg/{img}" for img in os.listdir(f"{tmp_dir}/used_seg")]
    # gen_depth = DepthImages(seg_img_paths, f"{tmp_dir}/depth", f"{tmp_dir}/color")
    # color_paths, depth_paths = gen_depth.generator()
    # pcl = PCL()
    # pcd = pcl.generate(color_paths, depth_paths)
    # o3d.io.write_point_cloud(f"{tmp_dir}/pcd.ply", pcd)
    ws = f"{tmp_dir}/used_seg/" if use_seg else tmp_dir
    pcl_gen = PCL(ws, tmp_dir)
    pcl_gen.generate()
    return ply_to_glb(f"{tmp_dir}/pcd.ply")

def mesh_gen(tmp_dir, use_seg):
    #TODO: call API to generate mesh
    mesh = trimesh.load(f"{tmp_dir}/mesh.ply")
    mesh.export(f"{tmp_dir}/mesh_normal.ply", file_type="ply")

    color_path = ply_to_glb(f"{tmp_dir}/mesh.ply")
    normal_path = ply_to_glb(f"{tmp_dir}/mesh_normal.ply")

    return color_path, normal_path


def feed_example_to_gallery(img):
    for display_img in display_imgs:
        display_img = display_img[0]
        diff = ImageChops.difference(img, display_img)
        if not diff.getbbox():  # two images are the same
            img_id = display_img.filename
            data_dir = os.path.join(data_folder, str(img_id))
            data_fns = os.listdir(data_dir)
            data_fns.sort()
            data_imgs = []
            for data_fn in data_fns:
                file_path = os.path.join(data_dir, data_fn)
                img = Image.open(file_path)
                data_imgs.append(img)
            return data_imgs
    return [img]


custom_theme = gr.themes.Soft(primary_hue="blue").set(
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
)

def run_demo(): 


    # Gradio blocks
    with gr.Blocks(title=_TITLE, css="style.css", theme=custom_theme) as demo:
        tmp_dir_unposed = gr.State("./demo_exp/placeholder")
        display_folder = os.path.join(os.path.dirname(__file__), "examples_display")
        os.makedirs("examples_display", exist_ok=True)
        display_fns = os.listdir(display_folder)
        display_fns.sort()
        display_imgs = []
        for i, display_fn in enumerate(display_fns):
            file_path = os.path.join(display_folder, display_fn)
            img = Image.open(file_path)
            img.filename = i
            display_imgs.append([img])
        data_folder = os.path.join(os.path.dirname(__file__), "examples_data")

        # UI
        with gr.Row():
            gr.Markdown("# " + _TITLE)
        with gr.Row():
            gr.Markdown("### " + _DESCRIPTION)
        with gr.Row():
            guide_text = gr.HTML(
                message("Input image(s) of object that you want to generate mesh with.")
            )
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=5):
                        input_gallery = gr.Gallery(
                            label="Input Images",
                            show_label=False,
                            columns=[3],
                            rows=[2],
                            object_fit="contain",
                            height=400,
                            show_share_button=False,
                        )
                        input_image = gr.Image(
                            type="pil",
                            image_mode="RGBA",
                            visible=False,
                        )
                    with gr.Column(scale=5):
                        processed_gallery = gr.Gallery(
                            label="Background Removal",
                            columns=[3],
                            rows=[2],
                            object_fit="contain",
                            height=400,
                            interactive=False,
                            show_share_button=False,
                        )
                with gr.Row():
                    with gr.Column(scale=5):
                        example = gr.Examples(
                            examples=display_imgs,
                            inputs=[input_image],
                            outputs=[input_gallery],
                            fn=feed_example_to_gallery,
                            label="Image Examples (Click one of the images below to start)",
                            examples_per_page=10,
                            run_on_click=True,
                        )
                    with gr.Column(scale=5):
                        with gr.Row():
                            bg_removed_checkbox = gr.Checkbox(
                                value=True,
                                label="Use background removed images (uncheck to use original)",
                                interactive=True,
                            )
                        with gr.Row():
                            run_btn = gr.Button(
                                "Generate",
                                variant="primary",
                                interactive=False,
                            )
                with gr.Row():
                    with gr.Column(scale=5):
                        pcl_output = gr.Model3D(
                            label="Generated Point Cloud",
                            elem_id="pcl-out",
                            height=400,
                        )
                # with gr.Row():
                #     with gr.Column(scale=5):
                #         mesh_output = Model3DColor(
                #             label="Generated Mesh (color)",
                #             elem_id="mesh-out",
                #             height=400,
                #         )
                #     with gr.Column(scale=5):
                #         mesh_output_normal = Model3DNormal(
                #             label="Generated Mesh (normal)",
                #             elem_id="mesh-normal-out",
                #             height=400,
                #         )

        # Callbacks
        generating_mesh = gr.State(False)
        disable_button = lambda: gr.Button(interactive=False)
        enable_button = lambda: gr.Button(interactive=True)
        update_guide = lambda GUIDE_TEXT, icon_type="info": gr.HTML(
            value=message(GUIDE_TEXT, icon_type)
        )

        def is_cleared(content):
            if content:
                raise ValueError  # gr.Error(visible=False) doesn't work, trick for not showing error message

        def not_cleared(content):
            if not content:
                raise ValueError

        def toggle_mesh_generation_status(generating_mesh):
            generating_mesh = not generating_mesh
            return generating_mesh

        def is_generating_mesh(generating_mesh):
            if generating_mesh:
                raise ValueError

        # Upload event listener for input gallery
        input_gallery.upload(
            fn=disable_button,
            outputs=[run_btn],
            queue=False,
        ).success(
            fn=create_tmp_dir,
            outputs=[tmp_dir_unposed],
            queue=True,
        ).success(
            fn=partial(
                update_guide, "Removing background of the input image(s)...", "wait"
            ),
            outputs=[guide_text],
            queue=False,
        ).success(
            fn=preprocess_imgs,
            inputs=[tmp_dir_unposed, input_gallery],
            outputs=[processed_gallery],
            queue=True,
        ).success(
            fn=partial(update_guide, "Click <b>Generate</b> to generate mesh.", "cursor"),
            outputs=[guide_text],
            queue=False,
        ).success(
            fn=is_generating_mesh,
            inputs=[generating_mesh],
            queue=False,
        ).success(
            fn=enable_button,
            outputs=[run_btn],
            queue=False,
        )

        # Clear event listener for input gallery
        input_gallery.change(
            fn=is_cleared,
            inputs=[input_gallery],
            queue=False,
        ).success(
            fn=disable_button,
            outputs=[run_btn],
            queue=False,
        ).success(
            fn=lambda: None,
            outputs=[input_image],
            queue=False,
        ).success(
            fn=lambda: None,
            outputs=[processed_gallery],
            queue=False,
        ).success(
            fn=partial(
                update_guide,
                "Input image(s) of object that you want to generate mesh with.",
                "info",
            ),
            outputs=[guide_text],
            queue=False,
        )

        # Change event listener for input image
        input_image.change(
            fn=not_cleared,
            inputs=[input_image],
            queue=False,
        ).success(
            fn=disable_button,
            outputs=run_btn,
            queue=False,
        ).success(
            fn=create_tmp_dir,
            outputs=tmp_dir_unposed,
            queue=True,
        ).success(
            fn=partial(
                update_guide, "Removing background of the input image(s)...", "wait"
            ),
            outputs=[guide_text],
            queue=False,
        ).success(
            fn=preprocess_imgs,
            inputs=[tmp_dir_unposed, input_gallery],
            outputs=[processed_gallery],
            queue=True,
        ).success(
            fn=partial(update_guide, "Click <b>Generate</b> to generate mesh.", "cursor"),
            outputs=[guide_text],
            queue=False,
        ).success(
            fn=is_generating_mesh,
            inputs=[generating_mesh],
            queue=False,
        ).success(
            fn=enable_button,
            outputs=run_btn,
            queue=False,
        )

        # Click event listener for run button
        run_btn.click(
            fn=disable_button,
            outputs=[run_btn],
            queue=False,
        ).success(
            fn=lambda: None,
            outputs=[pcl_output],
            queue=False,
        ).success(
            fn=partial(update_guide, "Generating the point cloud...", "wait"),
            outputs=[guide_text],
            queue=False,
        ).success(
            fn=pcd_gen,
            inputs=[tmp_dir_unposed, bg_removed_checkbox],
            outputs=[pcl_output],
            queue=True,
        ).success(
        # ).success(
        #     fn=lambda: None,
        #     outputs=[mesh_output],
        #     queue=False,
        # ).success(
        #     fn=lambda: None,
        #     outputs=[mesh_output_normal],
        #     queue=False,
        # ).success(
            fn=partial(update_guide, "Generating the mesh...", "wait"),
            outputs=[guide_text],
            queue=False,
        ).success(
            fn=toggle_mesh_generation_status,
            inputs=[generating_mesh],
            outputs=[generating_mesh],
            queue=False,
        ).success(
            fn=mesh_gen,
            inputs=[tmp_dir_unposed, bg_removed_checkbox],
            # outputs=[mesh_output, mesh_output_normal],
            queue=True,
        ).success(
            fn=toggle_mesh_generation_status,
            inputs=[generating_mesh],
            outputs=[generating_mesh],
            queue=False,
        ).success(
            fn=partial(
                update_guide,
                "Successfully generated the mesh. (It might take a few seconds to load the mesh)",
                "done",
            ),
            outputs=[guide_text],
            queue=False,
        ).success(
            fn=not_cleared,
            inputs=[input_gallery],
            queue=False,
        ).success(
            fn=enable_button,
            outputs=[run_btn],
            queue=False,
        )

    demo.launch(share=True)

if __name__ == "__main__":
    run_demo()