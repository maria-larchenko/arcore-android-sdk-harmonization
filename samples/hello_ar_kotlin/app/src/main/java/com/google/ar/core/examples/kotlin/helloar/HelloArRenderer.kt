/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ar.core.examples.kotlin.helloar

//import java.nio.FloatBuffer
import android.R.attr.height
import android.R.attr.width
import android.content.ContentValues
import android.graphics.Bitmap
import android.opengl.GLES30
import android.opengl.Matrix
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ar.core.Anchor
import com.google.ar.core.Camera
import com.google.ar.core.DepthPoint
import com.google.ar.core.Frame
import com.google.ar.core.InstantPlacementPoint
import com.google.ar.core.LightEstimate
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Session
import com.google.ar.core.Trackable
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper
import com.google.ar.core.examples.java.common.samplerender.Framebuffer
import com.google.ar.core.examples.java.common.samplerender.GLError
import com.google.ar.core.examples.java.common.samplerender.Mesh
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import com.google.ar.core.examples.java.common.samplerender.Shader
import com.google.ar.core.examples.java.common.samplerender.Texture
import com.google.ar.core.examples.java.common.samplerender.VertexBuffer
import com.google.ar.core.examples.java.common.samplerender.arcore.BackgroundRenderer
import com.google.ar.core.examples.java.common.samplerender.arcore.PlaneRenderer
import com.google.ar.core.examples.java.common.samplerender.arcore.SpecularCubemapFilter
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder


/** Renders the HelloAR application using our example Renderer. */
class HelloArRenderer(val activity: HelloArActivity) :
  SampleRender.Renderer, DefaultLifecycleObserver {
  companion object {
    val TAG = "HelloArRenderer"

    // See the definition of updateSphericalHarmonicsCoefficients for an explanation of these
    // constants.
    private val sphericalHarmonicFactors =
      floatArrayOf(
        0.282095f,
        -0.325735f,
        0.325735f,
        -0.325735f,
        0.273137f,
        -0.273137f,
        0.078848f,
        -0.273137f,
        0.136569f
      )

    private val Z_NEAR = 0.1f
    private val Z_FAR = 100f

    // Assumed distance from the device camera to the surface on which user will try to place
    // objects.
    // This value affects the apparent scale of objects while the tracking method of the
    // Instant Placement point is SCREENSPACE_WITH_APPROXIMATE_DISTANCE.
    // Values in the [0.2, 2.0] meter range are a good choice for most AR experiences. Use lower
    // values for AR experiences where users are expected to place objects on surfaces close to the
    // camera. Use larger values for experiences where the user will likely be standing and trying
    // to
    // place an object on the ground or floor in front of them.
    val APPROXIMATE_DISTANCE_METERS = 2.0f

    val CUBEMAP_RESOLUTION = 16
    val CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES = 32
  }

  lateinit var render: SampleRender
  lateinit var planeRenderer: PlaneRenderer
  lateinit var backgroundRenderer: BackgroundRenderer
  lateinit var virtualSceneFramebuffer: Framebuffer
  lateinit var virtualMaskFramebuffer: Framebuffer
  lateinit var compositeFramebuffer: Framebuffer
  // Full-screen quad used for post-processing passes such as color+mask composition
  lateinit var compositeMesh: Mesh
  var hasSetTextureNames = false

  // Point Cloud
  lateinit var pointCloudVertexBuffer: VertexBuffer
  lateinit var pointCloudMesh: Mesh
  lateinit var pointCloudShader: Shader

  // Keep track of the last point cloud rendered to avoid updating the VBO if point cloud
  // was not changed.  Do this using the timestamp since we can't compare PointCloud objects.
  var lastPointCloudTimestamp: Long = 0

  // lateinit var interpreter: Interpreter
  lateinit var compiledModel: CompiledModel

  // Virtual object (ARCore pawn)
  lateinit var virtualObjectMesh: Mesh
  lateinit var virtualObjectShader: Shader
  lateinit var maskObjectShader: Shader
  lateinit var compositeShader: Shader
  lateinit var virtualObjectAlbedoTexture: Texture
  lateinit var virtualObjectAlbedoInstantPlacementTexture: Texture

  private val wrappedAnchors = mutableListOf<WrappedAnchor>()

  // Environmental HDR
  lateinit var dfgTexture: Texture
  lateinit var cubemapFilter: SpecularCubemapFilter

  // Temporary matrix allocated here to reduce number of allocations for each frame.
  val modelMatrix = FloatArray(16)
  val viewMatrix = FloatArray(16)
  val projectionMatrix = FloatArray(16)
  val modelViewMatrix = FloatArray(16) // view x model

  val modelViewProjectionMatrix = FloatArray(16) // projection x view x model

  val sphericalHarmonicsCoefficients = FloatArray(9 * 3)
  val viewInverseMatrix = FloatArray(16)
  val worldLightDirection = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
  val viewLightDirection = FloatArray(4) // view x world light direction

  val session
    get() = activity.arCoreSessionHelper.session

  val displayRotationHelper = DisplayRotationHelper(activity)
  val trackingStateHelper = TrackingStateHelper(activity)

  override fun onResume(owner: LifecycleOwner) {
    displayRotationHelper.onResume()
    hasSetTextureNames = false
  }

  override fun onPause(owner: LifecycleOwner) {
    displayRotationHelper.onPause()
  }

  override fun onSurfaceCreated(render: SampleRender) {
    // --------  Init the encoder model
     compiledModel = CompiledModel.create(
        activity.assets,
        "mkl_encoder_256.tflite",
        CompiledModel.Options(Accelerator.GPU)
    )
    Log.d(TAG, compiledModel.toString())

    // Prepare the rendering objects.
    // This involves reading shaders and 3D model files, so may throw an IOException.
    try {
      planeRenderer = PlaneRenderer(render)
      backgroundRenderer = BackgroundRenderer(render)
      virtualSceneFramebuffer = Framebuffer(render, /*width=*/ 1, /*height=*/ 1)
      virtualMaskFramebuffer = Framebuffer(render, /*width=*/ 1, /*height=*/ 1)
      compositeFramebuffer = Framebuffer(render, /*width=*/ 1, /*height=*/ 1)

      cubemapFilter =
        SpecularCubemapFilter(render, CUBEMAP_RESOLUTION, CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES)
      // Load environmental lighting values lookup table
      dfgTexture =
        Texture(
          render,
          Texture.Target.TEXTURE_2D,
          Texture.WrapMode.CLAMP_TO_EDGE,
          /*useMipmaps=*/ false
        )
      // The dfg.raw file is a raw half-float texture with two channels.
      val dfgResolution = 64
      val dfgChannels = 2
      val halfFloatSize = 2

      val buffer: ByteBuffer =
        ByteBuffer.allocateDirect(dfgResolution * dfgResolution * dfgChannels * halfFloatSize)
      activity.assets.open("models/dfg.raw").use { it.read(buffer.array()) }

      // SampleRender abstraction leaks here.
      GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, dfgTexture.textureId)
      GLError.maybeThrowGLException("Failed to bind DFG texture", "glBindTexture")
      GLES30.glTexImage2D(
        GLES30.GL_TEXTURE_2D,
        /*level=*/ 0,
        GLES30.GL_RG16F,
        /*width=*/ dfgResolution,
        /*height=*/ dfgResolution,
        /*border=*/ 0,
        GLES30.GL_RG,
        GLES30.GL_HALF_FLOAT,
        buffer
      )
      GLError.maybeThrowGLException("Failed to populate DFG texture", "glTexImage2D")

      // Point cloud
      pointCloudShader =
        Shader.createFromAssets(
            render,
            "shaders/point_cloud.vert",
            "shaders/point_cloud.frag",
            /*defines=*/ null
          )
          .setVec4("u_Color", floatArrayOf(31.0f / 255.0f, 188.0f / 255.0f, 210.0f / 255.0f, 1.0f))
          .setFloat("u_PointSize", 5.0f)

      // four entries per vertex: X, Y, Z, confidence
      pointCloudVertexBuffer =
        VertexBuffer(render, /*numberOfEntriesPerVertex=*/ 4, /*entries=*/ null)
      val pointCloudVertexBuffers = arrayOf(pointCloudVertexBuffer)
      pointCloudMesh =
        Mesh(render, Mesh.PrimitiveMode.POINTS, /*indexBuffer=*/ null, pointCloudVertexBuffers)

      // Virtual object to render (ARCore pawn)
//      val myobject = "ARCore pawn"
      val myobject = "Minecraft chicken"

      if (myobject == "ARCore pawn") {
        virtualObjectAlbedoTexture =
          Texture.createFromAsset(
            render,
            "models/pawn_albedo.png",
            Texture.WrapMode.CLAMP_TO_EDGE,
            Texture.ColorFormat.SRGB
          )
        virtualObjectAlbedoInstantPlacementTexture =
          Texture.createFromAsset(
            render,
            "models/pawn_albedo_instant_placement.png",
            Texture.WrapMode.CLAMP_TO_EDGE,
            Texture.ColorFormat.SRGB
          )
        val virtualObjectPbrTexture =
          Texture.createFromAsset(
            render,
            "models/pawn_roughness_metallic_ao.png",
            Texture.WrapMode.CLAMP_TO_EDGE,
            Texture.ColorFormat.LINEAR
          )
        virtualObjectMesh = Mesh.createFromAsset(render, "models/pawn.obj")
        virtualObjectShader =
          Shader.createFromAssets(
            render,
            "shaders/environmental_hdr.vert",
            "shaders/environmental_hdr.frag",
            mapOf("NUMBER_OF_MIPMAP_LEVELS" to cubemapFilter.numberOfMipmapLevels.toString())
          )
            .setTexture("u_AlbedoTexture", virtualObjectAlbedoTexture)
            .setTexture("u_RoughnessMetallicAmbientOcclusionTexture", virtualObjectPbrTexture)
            .setTexture("u_Cubemap", cubemapFilter.filteredCubemapTexture)
            .setTexture("u_DfgTexture", dfgTexture)
        maskObjectShader =
          Shader.createFromAssets(
            render,
            "shaders/environmental_hdr.vert",
            "shaders/mask_hdr.frag",
            mapOf("NUMBER_OF_MIPMAP_LEVELS" to cubemapFilter.numberOfMipmapLevels.toString())
          )
      } else {
        virtualObjectAlbedoTexture =
          Texture.createFromAsset(
            render,
            "models/chair_albedo.png",
//            "models/chicken_albedo.png",
            Texture.WrapMode.CLAMP_TO_EDGE,
            Texture.ColorFormat.SRGB
          )
        virtualObjectMesh = Mesh.createFromAsset(render, "models/chair.obj")
//        virtualObjectMesh = Mesh.createFromAsset(render, "models/chicken.obj")
        virtualObjectShader =
          Shader.createFromAssets(
            render,
            "shaders/environmental_hdr.vert",
            "shaders/environmental_hdr.frag",
            mapOf("NUMBER_OF_MIPMAP_LEVELS" to cubemapFilter.numberOfMipmapLevels.toString())
          )
            .setTexture("u_AlbedoTexture", virtualObjectAlbedoTexture)
            .setTexture("u_Cubemap", cubemapFilter.filteredCubemapTexture)
            .setTexture("u_DfgTexture", dfgTexture)
        maskObjectShader =
          Shader.createFromAssets(
            render,
            "shaders/environmental_hdr.vert",
            "shaders/mask_hdr.frag",
            mapOf("NUMBER_OF_MIPMAP_LEVELS" to cubemapFilter.numberOfMipmapLevels.toString())
          )
      }
      compositeShader = Shader.createFromAssets(
        render,
        "shaders/composite_rgba.vert",
        "shaders/composite_rgba.frag",
        null)

      // ---------- build a full-screen quad mesh (NDC positions + texcoords) ---------
      val coords = floatArrayOf(
        -1f, -1f,   // bottom-left
         1f, -1f,   // bottom-right
        -1f,  1f,   // top-left
         1f,  1f    // top-right
      )
      val tex = floatArrayOf(
          0f, 0f,
          1f, 0f,
          0f, 1f,
          1f, 1f
      )
      val coordsBuffer = ByteBuffer.allocateDirect(coords.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(coords)
      coordsBuffer.rewind()
      val texBuffer = ByteBuffer.allocateDirect(tex.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(tex)
      texBuffer.rewind()

      val screenVb = VertexBuffer(render, /*entriesPerVertex=*/2, coordsBuffer)
      val texVb    = VertexBuffer(render, /*entriesPerVertex=*/2, texBuffer)
      compositeMesh = Mesh(render, Mesh.PrimitiveMode.TRIANGLE_STRIP, null, arrayOf(screenVb, texVb))
    } catch (e: IOException) {
      Log.e(TAG, "Failed to read a required asset file", e)
      showError("Failed to read a required asset file: $e")
    }
  }

  override fun onSurfaceChanged(render: SampleRender, width: Int, height: Int) {
    displayRotationHelper.onSurfaceChanged(width, height)
    virtualSceneFramebuffer.resize(width, height)
    virtualMaskFramebuffer.resize(width, height)
    compositeFramebuffer.resize(256, 256)
  }

  private var saveNextFrame = false

  fun requestSave() {             // call this from a button in the UI
      saveNextFrame = true
  }

  override fun onDrawFrame(render: SampleRender) {
    val session = session ?: return

    // Texture names should only be set once on a GL thread unless they change. This is done during
    // onDrawFrame rather than onSurfaceCreated since the session is not guaranteed to have been
    // initialized during the execution of onSurfaceCreated.
    if (!hasSetTextureNames) {
      session.setCameraTextureNames(intArrayOf(backgroundRenderer.cameraColorTexture.textureId))
      hasSetTextureNames = true
    }

    // -- Update per-frame state

    // Notify ARCore session that the view size changed so that the perspective matrix and
    // the video background can be properly adjusted.
    displayRotationHelper.updateSessionIfNeeded(session)

    // Obtain the current frame from ARSession. When the configuration is set to
    // UpdateMode.BLOCKING (it is by default), this will throttle the rendering to the
    // camera framerate.
    val frame =
      try {
        session.update()
      } catch (e: CameraNotAvailableException) {
        Log.e(TAG, "Camera not available during onDrawFrame", e)
        showError("Camera not available. Try restarting the app.")
        return
      }

    val camera = frame.camera

    // Update BackgroundRenderer state to match the depth settings.
    try {
      backgroundRenderer.setUseDepthVisualization(
        render,
        activity.depthSettings.depthColorVisualizationEnabled()
      )
      backgroundRenderer.setUseOcclusion(render, activity.depthSettings.useDepthForOcclusion())
    } catch (e: IOException) {
      Log.e(TAG, "Failed to read a required asset file", e)
      showError("Failed to read a required asset file: $e")
      return
    }

    // BackgroundRenderer.updateDisplayGeometry must be called every frame to update the coordinates
    // used to draw the background camera image.
    backgroundRenderer.updateDisplayGeometry(frame)
    val shouldGetDepthImage =
      activity.depthSettings.useDepthForOcclusion() ||
        activity.depthSettings.depthColorVisualizationEnabled()
    if (camera.trackingState == TrackingState.TRACKING && shouldGetDepthImage) {
      try {
        val depthImage = frame.acquireDepthImage16Bits()
        backgroundRenderer.updateCameraDepthTexture(depthImage)
        depthImage.close()
      } catch (e: NotYetAvailableException) {
        // This normally means that depth data is not available yet. This is normal so we will not
        // spam the logcat with this.
      }
    }

    // Handle one tap per frame.
    handleTap(frame, camera)

    // Keep the screen unlocked while tracking, but allow it to lock when tracking stops.
    trackingStateHelper.updateKeepScreenOnFlag(camera.trackingState)

    // Show a message based on whether tracking has failed, if planes are detected, and if the user
    // has placed any objects.
    val message: String? =
      when {
        camera.trackingState == TrackingState.PAUSED &&
          camera.trackingFailureReason == TrackingFailureReason.NONE ->
          activity.getString(R.string.searching_planes)
        camera.trackingState == TrackingState.PAUSED ->
          TrackingStateHelper.getTrackingFailureReasonString(camera)
        session.hasTrackingPlane() && wrappedAnchors.isEmpty() ->
          activity.getString(R.string.waiting_taps)
        session.hasTrackingPlane() && wrappedAnchors.isNotEmpty() -> null
        else -> activity.getString(R.string.searching_planes)
      }
    if (message == null) {
      activity.view.snackbarHelper.hide(activity)
    } else {
      activity.view.snackbarHelper.showMessage(activity, message)
    }

    // -- Draw background
    if (frame.timestamp != 0L) {
      // Suppress rendering if the camera did not produce the first frame yet. This is to avoid
      // drawing possible leftover data from previous sessions if the texture is reused.
      backgroundRenderer.drawBackground(render)
    }

    // If not tracking, don't draw 3D objects.
    if (camera.trackingState == TrackingState.PAUSED) {
      return
    }

    // -- Draw non-occluded virtual objects (planes, point cloud)

    // Get projection matrix.
    camera.getProjectionMatrix(projectionMatrix, 0, Z_NEAR, Z_FAR)

    // Get camera matrix and draw.
    camera.getViewMatrix(viewMatrix, 0)
    frame.acquirePointCloud().use { pointCloud ->
      if (pointCloud.timestamp > lastPointCloudTimestamp) {
        pointCloudVertexBuffer.set(pointCloud.points)
        lastPointCloudTimestamp = pointCloud.timestamp
      }
      Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0)
      pointCloudShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix)
      render.draw(pointCloudMesh, pointCloudShader)
    }

    // Visualize planes.
    planeRenderer.drawPlanes(
      render,
      session.getAllTrackables<Plane>(Plane::class.java),
      camera.displayOrientedPose,
      projectionMatrix,
    )

    // -- Draw occluded virtual objects

    // Update lighting parameters in the shader
    updateLightEstimation(frame.lightEstimate, viewMatrix)

    // Visualize anchors created by touch.
    render.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f)
    render.clear(virtualMaskFramebuffer, 0f, 0f, 0f, 0f)
    render.clear(compositeFramebuffer, 0f, 0f, 0f, 0f)
    for ((anchor, trackable) in
      wrappedAnchors.filter { it.anchor.trackingState == TrackingState.TRACKING }) {
      // Get the current pose of an Anchor in world space. The Anchor pose is updated
      // during calls to session.update() as ARCore refines its estimate of the world.
      anchor.pose.toMatrix(modelMatrix, 0)

      // Calculate model/view/projection matrices
      Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0)
      Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0)

      val testMat = FloatArray(16)
      testMat.fill(0.0f)
      testMat[0] = 1.0f
      testMat[5] = 1.0f
      testMat[10] = 1.0f
      testMat[15] = 1.0f

      // Update shader properties and draw
      virtualObjectShader.setMat4("u_ModelView", modelViewMatrix)
      virtualObjectShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix)
      virtualObjectShader.setMat4("u_ColorCorrection", testMat)
      val texture =
        if ((trackable as? InstantPlacementPoint)?.trackingMethod ==
            InstantPlacementPoint.TrackingMethod.SCREENSPACE_WITH_APPROXIMATE_DISTANCE
        ) {
          virtualObjectAlbedoInstantPlacementTexture
        } else {
          virtualObjectAlbedoTexture
        }
      virtualObjectShader.setTexture("u_AlbedoTexture", texture)
      render.draw(virtualObjectMesh, virtualObjectShader)

      backgroundRenderer.drawBackground(render, virtualSceneFramebuffer)
      render.draw(virtualObjectMesh, virtualObjectShader, virtualSceneFramebuffer)

      // Update mask shader properties and draw
      maskObjectShader.setMat4("u_ModelView", modelViewMatrix)
      maskObjectShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix)
      render.draw(virtualObjectMesh, maskObjectShader, virtualMaskFramebuffer)

      // Update composite mask and image shaders properties and draw
      compositeShader
        .setTexture("u_ColorTex", virtualSceneFramebuffer.colorTexture)
        .setTexture("u_MaskTex",  virtualMaskFramebuffer.colorTexture)
      render.draw(compositeMesh, compositeShader, compositeFramebuffer)
    }

    if (saveNextFrame) {
        saveNextFrame = false
        saveImages(frame)       // <— background + foreground
    }

    // Compose the virtual scene with the background.
    backgroundRenderer.drawVirtualScene(render, virtualSceneFramebuffer, Z_NEAR, Z_FAR) //draws a mask over the object
  }

  /** Checks if we detected at least one plane. */
  private fun Session.hasTrackingPlane() =
    getAllTrackables(Plane::class.java).any { it.trackingState == TrackingState.TRACKING }

  /** Update state based on the current frame's light estimation. */
  private fun updateLightEstimation(lightEstimate: LightEstimate, viewMatrix: FloatArray) {
    if (lightEstimate.state != LightEstimate.State.VALID) {
      virtualObjectShader.setBool("u_LightEstimateIsValid", false)
      return
    }
    virtualObjectShader.setBool("u_LightEstimateIsValid", true)
    Matrix.invertM(viewInverseMatrix, 0, viewMatrix, 0)
    virtualObjectShader.setMat4("u_ViewInverse", viewInverseMatrix)
    updateMainLight(
      lightEstimate.environmentalHdrMainLightDirection,
      lightEstimate.environmentalHdrMainLightIntensity,
      viewMatrix
    )
    updateSphericalHarmonicsCoefficients(lightEstimate.environmentalHdrAmbientSphericalHarmonics)
    cubemapFilter.update(lightEstimate.acquireEnvironmentalHdrCubeMap())
  }

  private fun updateMainLight(
    direction: FloatArray,
    intensity: FloatArray,
    viewMatrix: FloatArray
  ) {
    // We need the direction in a vec4 with 0.0 as the final component to transform it to view space
    worldLightDirection[0] = direction[0]
    worldLightDirection[1] = direction[1]
    worldLightDirection[2] = direction[2]
    Matrix.multiplyMV(viewLightDirection, 0, viewMatrix, 0, worldLightDirection, 0)
    virtualObjectShader.setVec4("u_ViewLightDirection", viewLightDirection)
    virtualObjectShader.setVec3("u_LightIntensity", intensity)
  }

  private fun updateSphericalHarmonicsCoefficients(coefficients: FloatArray) {
    // Pre-multiply the spherical harmonics coefficients before passing them to the shader. The
    // constants in sphericalHarmonicFactors were derived from three terms:
    //
    // 1. The normalized spherical harmonics basis functions (y_lm)
    //
    // 2. The lambertian diffuse BRDF factor (1/pi)
    //
    // 3. A <cos> convolution. This is done to so that the resulting function outputs the irradiance
    // of all incoming light over a hemisphere for a given surface normal, which is what the shader
    // (environmental_hdr.frag) expects.
    //
    // You can read more details about the math here:
    // https://google.github.io/filament/Filament.html#annex/sphericalharmonics
    require(coefficients.size == 9 * 3) {
      "The given coefficients array must be of length 27 (3 components per 9 coefficients"
    }

    // Apply each factor to every component of each coefficient
    for (i in 0 until 9 * 3) {
      sphericalHarmonicsCoefficients[i] = coefficients[i] * sphericalHarmonicFactors[i / 3]
    }
    virtualObjectShader.setVec3Array(
      "u_SphericalHarmonicsCoefficients",
      sphericalHarmonicsCoefficients
    )
  }

  // Handle only one tap per frame, as taps are usually low frequency compared to frame rate.
  private fun handleTap(frame: Frame, camera: Camera) {
    if (camera.trackingState != TrackingState.TRACKING) return
    val tap = activity.view.tapHelper.poll() ?: return

    val hitResultList =
      if (activity.instantPlacementSettings.isInstantPlacementEnabled) {
        frame.hitTestInstantPlacement(tap.x, tap.y, APPROXIMATE_DISTANCE_METERS)
      } else {
        frame.hitTest(tap)
      }

    // Hits are sorted by depth. Consider only closest hit on a plane, Oriented Point, Depth Point,
    // or Instant Placement Point.
    val firstHitResult =
      hitResultList.firstOrNull { hit ->
        when (val trackable = hit.trackable!!) {
          is Plane ->
            trackable.isPoseInPolygon(hit.hitPose) &&
              PlaneRenderer.calculateDistanceToPlane(hit.hitPose, camera.pose) > 0
          is Point -> trackable.orientationMode == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL
          is InstantPlacementPoint -> true
          // DepthPoints are only returned if Config.DepthMode is set to AUTOMATIC.
          is DepthPoint -> true
          else -> false
        }
      }

    if (firstHitResult != null) {
      // If we already have an anchor, remove it and create a new one at the new location
      // This effectively moves the single object to the new position
      if (wrappedAnchors.isNotEmpty()) {
        wrappedAnchors[0].anchor.detach()
        wrappedAnchors.clear()
      }

      // Adding an Anchor tells ARCore that it should track this position in
      // space. This anchor is created on the Plane to place the 3D model
      // in the correct position relative both to the world and to the plane.
      wrappedAnchors.add(WrappedAnchor(firstHitResult.createAnchor(), firstHitResult.trackable))

      // For devices that support the Depth API, shows a dialog to suggest enabling
      // depth-based occlusion. This dialog needs to be spawned on the UI thread.
      activity.runOnUiThread { activity.view.showOcclusionDialogIfNeeded() }
    }
  }

  private fun showError(errorMessage: String) =
    activity.view.snackbarHelper.showError(activity, errorMessage)

  private fun saveImages(frame: Frame) {
      val resolver = activity.contentResolver
      val timestamp = System.currentTimeMillis()

      // --- 1. virtual scene (PNG) ----------------------------------------
      val w = virtualSceneFramebuffer.width
      val h = virtualSceneFramebuffer.height

//      val buf = IntArray(w * h)   // produces wrong conversion of color schemes!
//      GLES30.glReadPixels(0, 0, w, h,
//          GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, IntBuffer.wrap(buf))

      // Check which framebuffer is in use:
      val bound = IntArray(1)
      // Generic binding (what you normally care about)
      GLES30.glGetIntegerv(GLES30.GL_FRAMEBUFFER_BINDING, bound, 0)
      // If you used separate read/draw bindings:
      GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, bound, 0)
      GLES30.glGetIntegerv(GLES30.GL_READ_FRAMEBUFFER_BINDING, bound, 0)
      Log.d(TAG, "Currently bound FB = ${bound[0]}")  // 0 means the default window framebuffer
      Log.d(TAG, "Draw FB = ${bound[0]}")
      Log.d(TAG, "Read FB = ${bound[0]}")
      Log.d(TAG, "virtualSceneFramebuffer ID = ${virtualSceneFramebuffer.framebufferId}")

      val byteBuf = ByteBuffer.allocateDirect(w * h * 4)
      GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, virtualSceneFramebuffer.framebufferId)
      GLES30.glReadPixels(0, 0, w, h, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, byteBuf)

      // https://stackoverflow.com/questions/16461284/difference-between-bytebuffer-flip-and-bytebuffer-rewind
      byteBuf.rewind()
      val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
      bmp.copyPixelsFromBuffer(byteBuf)          // copies bytes as R-G-B-A

      val pngValues = ContentValues().apply {
          put(MediaStore.Images.Media.DISPLAY_NAME, "foreground_${timestamp}.png")
          put(MediaStore.Images.Media.MIME_TYPE, "image/png")
          put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ARCore")
      }
      val pngUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, pngValues)
      if (pngUri == null) {
          Log.e(TAG, "Failed to create MediaStore entry for PNG file")
      } else {
          resolver.openOutputStream(pngUri)?.use { os ->
              bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
              os.flush()
              Log.d(TAG, "Saved foreground PNG to ${pngUri}")
          }
      }

      // --- 2. virtual object mask (PNG) ----------------------------------------
      byteBuf.clear()
      GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, virtualMaskFramebuffer.framebufferId)
      GLES30.glReadPixels(0, 0, w, h, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, byteBuf)
      byteBuf.rewind()
      bmp.copyPixelsFromBuffer(byteBuf)          // copies bytes as R-G-B-A

      val maskValues = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "mask_${timestamp}.png")
        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ARCore")
      }
      val maskUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, maskValues)
      if (maskUri == null) {
        Log.e(TAG, "Failed to create MediaStore entry for PNG file")
      } else {
        resolver.openOutputStream(maskUri)?.use { os ->
          bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
          os.flush()
          Log.d(TAG, "Saved mask PNG to ${maskUri}")
        }
      }

      // --- 3. encoder input  in RBGA format (PNG) debug ----------------------------------------
      val debugByteBuf = ByteBuffer.allocateDirect(256 * 256 * 4)
      GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, compositeFramebuffer.framebufferId)
      GLES30.glReadPixels(0, 0, 256, 256, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, debugByteBuf)

      // https://stackoverflow.com/questions/16461284/difference-between-bytebuffer-flip-and-bytebuffer-rewind
      debugByteBuf.rewind()
      val debugBmp = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
      debugBmp.copyPixelsFromBuffer(debugByteBuf)          // copies bytes as R-G-B-A

      val debugValues = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "debug_${timestamp}.png")
        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ARCore")
      }
      val debugUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, debugValues)
      if (debugUri == null) {
        Log.e(TAG, "Failed to create MediaStore entry for PNG file")
      } else {
        resolver.openOutputStream(debugUri)?.use { os ->
          debugBmp.compress(Bitmap.CompressFormat.PNG, 100, os)
          os.flush()
          Log.d(TAG, "Saved debug PNG to ${debugUri}")
        }
      }

//      // --- 4. background colour buffer (matches viewport/FBO) ------------
//      byteBuf.clear()
//      GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
//      GLES30.glReadPixels(0, 0, w, h, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, byteBuf)
//      byteBuf.rewind()
//      bmp.copyPixelsFromBuffer(byteBuf)          // copies bytes as R-G-B-A
//
//      val bgValues = ContentValues().apply {
//          put(MediaStore.Images.Media.DISPLAY_NAME, "background_${timestamp}.jpg")
//          put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
//          put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ARCore")
//      }
//
//      val bgUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, bgValues)
//      if (bgUri == null) {
//          Log.e(TAG, "Failed to create MediaStore entry for background JPEG")
//      } else {
//          resolver.openOutputStream(bgUri)?.use { os ->
////              Bitmap.createBitmap(bgBuf, wBg, hBg, Bitmap.Config.ARGB_8888)
//              bmp.compress(Bitmap.CompressFormat.JPEG, 100, os)
//              os.flush()
//              Log.d(TAG, "Saved background JPEG to ${bgUri}")
//          }
//      }
  }


}

/**
 * Associates an Anchor with the trackable it was attached to. This is used to be able to check
 * whether or not an Anchor originally was attached to an {@link InstantPlacementPoint}.
 */
private data class WrappedAnchor(
  val anchor: Anchor,
  val trackable: Trackable,
)
