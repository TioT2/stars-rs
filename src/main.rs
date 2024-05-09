use wasm_bindgen::prelude::*;

#[derive(Copy, Clone)]
struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 {x, y, z}
    }
}

impl std::ops::Add<Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Self::Output {
        Self::Output {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z
        }
    }
}

impl std::ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z
        }
    }
}

impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z
        }
    }
}

struct IVec2 {
    pub x: i32,
    pub y: i32,
}

struct XorshiftRand {
    state: u32
}

impl XorshiftRand {
    pub fn new(seed: u32) -> XorshiftRand {
        XorshiftRand { state: seed }
    }

    pub fn rand_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;

        self.state
    }

    pub fn rand_unit_f32(&mut self) -> f32 {
        let u = self.rand_u32();

        (u as f32) / (u32::MAX as f32)
    }

    pub fn rand_f32(&mut self, begin: f32, end: f32) -> f32 {
        self.rand_unit_f32() * (end - begin) + begin
    }
}

#[derive(Copy, Clone)]
struct Frustum {
    pub width: f32,
    pub height: f32,
    pub far: f32,
    pub near: f32,
}

impl Frustum {
    pub fn new() -> Self {
        Self {
            width: 500.0,
            height: 500.0,
            far: 10000.0,
            near: 100.0,
        }
    }

    pub fn check_point_visibility(&self, point: &Vec3) -> bool {
        /* Check for near-far clip */
        point.z >= self.near &&
        point.z <= self.far &&

        /* Checking for cone in XZ plane */
        point.z >= 2.0 * self.near * point.x.abs() / self.width &&

        /* Checking for cone in YZ plane */
        point.z >= 2.0 * self.near * point.y.abs() / self.height
    }

    // Random visible star generation function
    pub fn gen_rand_visible(&self, rand: &mut XorshiftRand) -> Vec3 {
        let zu = 1.0 - rand.rand_unit_f32().powi(4);
        let z = zu * (self.far - self.near - f32::EPSILON * 2.0) + self.near + f32::EPSILON;
        let mx = z * self.width / (2.0 * self.near);
        let my = z * self.height / (2.0 * self.near);

        Vec3 {
            x: rand.rand_f32(-mx, mx),
            y: rand.rand_f32(-my, my),
            z,
        }
    }
}

struct Stars {
    rand: XorshiftRand,
    surface_extent: IVec2,
    stars: Vec<Vec3>,
    speed: f32,
    surface_data: std::cell::RefCell<Vec<u32>>,
    pub frustum: Frustum,
}

impl Stars {
    pub fn new() -> Stars {
        Stars {
            rand: XorshiftRand::new(47),
            surface_extent: IVec2 {x: 102, y: 108},
            stars: Vec::<Vec3>::new(),
            speed: 3000.0,
            surface_data: std::cell::RefCell::new({
                let mut stars = Vec::<u32>::new();

                stars.resize(192 * 108, 0);

                stars
            }),
            frustum: Frustum::new(),
        }
    }

    pub fn resize_stars(&mut self, new_star_count: usize) {
        let mut new_stars = Vec::<Vec3>::with_capacity(new_star_count);
        new_stars.resize(new_star_count, Vec3::new(0.0, 0.0, 0.0));

        for star in &mut new_stars {
            *star = self.frustum.gen_rand_visible(&mut self.rand);
        }
        // initialize new star

        self.stars = new_stars;
   }

    pub fn response(&mut self, dt: f32) {
        let vdt = self.speed * dt;

        /* Move every star */
        for star in &mut self.stars {
            star.z -= vdt;

            // Rebuild star in case it isn't visible
            if !self.frustum.check_point_visibility(star) {
                *star = self.frustum.gen_rand_visible(&mut self.rand);
            }
        }
    }

    pub fn translate(&mut self, vector: Vec3) {
        for star in &mut self.stars {
            *star = *star - vector;
        }
    }

    pub fn rotate_y(&mut self, angle: f32) {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        for star in &mut self.stars {
            *star = Vec3 {
                x: star.x * cos_angle - star.z * sin_angle,
                y: star.y,
                z: star.x * sin_angle + star.z * cos_angle,
            };
        }
    }

    pub fn render(&self, dst_bits: &mut [u32]) {
        let mut surface_data = self.surface_data.borrow_mut();

        unsafe {
            let data = surface_data.as_mut_ptr();

            let max_star_size: f32 = 4.0;
            let ex = self.surface_extent.x as f32 - max_star_size - 1.0;
            let ey = self.surface_extent.y as f32 - max_star_size - 1.0;

            let xm = self.frustum.near / self.frustum.width * ex;
            let xa = ex * 0.5 + 1.0;

            let ym = self.frustum.near / self.frustum.height * ey;
            let ya = ey * 0.5 + 1.0;

            let z_color_m = 255.0 / (self.frustum.far - self.frustum.near);
            let z_size_m = 1.0 / (self.frustum.far - self.frustum.near);

            for star in &self.stars {
                // Calculate projection of star into frustum
                let xp = (star.x / star.z * xm + xa) as usize;
                let yp = (star.y / star.z * ym + ya) as usize;
                let color: u32 = ((self.frustum.far - star.z) * z_color_m) as u32;
                let color = 0xFF000000 | color | (color << 8) | color << 16;
                let size = (self.frustum.far - star.z) * z_size_m;
                let size = (size * size * size * max_star_size) as usize;

                // *data.add(yp * self.surface_extent.x as usize + xp) = color;

                let xmax = xp + size;
                let ymax = yp + size;
                let mut y = yp;
                while y <= ymax {
                    let mut x = xp;
                    while x <= xmax {
                        *data.add(y * self.surface_extent.x as usize + x) = color;
                        x += 1;
                    }
                    y += 1;
                }
            }

            dst_bits.copy_from_slice(&surface_data);

            for star in &self.stars {
                // Calculate projection of star into frustum
                let xp = (star.x / star.z * xm + xa) as usize;
                let yp = (star.y / star.z * ym + ya) as usize;
                let size = (self.frustum.far - star.z) * z_size_m;
                let size = (size * size * size * max_star_size) as usize;

                // *data.add(yp * self.surface_extent.x as usize + xp) = color;

                let xmax = xp + size;
                let ymax = yp + size;
                let mut y = yp;
                while y <= ymax {
                    let mut x = xp;
                    while x <= xmax {
                        *data.add(y * self.surface_extent.x as usize + x) = 0xFF000000;
                        x += 1;
                    }
                    y += 1;
                }
            }
        }
    }

    pub fn resize_surface(&mut self, new_extent: IVec2) {
        self.surface_data.get_mut().resize((new_extent.x * new_extent.y).unsigned_abs() as usize, 0);
        self.surface_extent = new_extent;
    }
}

struct Timer {
    start_time_point: std::time::Instant,
    time_point: std::time::Instant,
    fps_time_point: std::time::Instant,
    time: f32,
    delta_time: f32,
    fps: f32,
    fps_counter: u32,
    fps_duration: f32,
}

impl Timer {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            start_time_point: now.clone(),
            time_point: now.clone(),
            fps_time_point: now.clone(),
            time: 0.0,
            delta_time: 0.01,
            fps: 30.0,
            fps_counter: 0,
            fps_duration: 3.0,
        }
    }

    pub fn response(&mut self) {
        let now = std::time::Instant::now();

        self.time = (now - self.start_time_point).as_secs_f32();
        self.delta_time = (now - self.time_point).as_secs_f32();


        self.fps_counter += 1;

        let fps_duration = (now - self.fps_time_point).as_secs_f32();
        if fps_duration >= self.fps_duration {
            self.fps = self.fps_counter as f32 / fps_duration;
            self.fps_time_point = now;
            self.fps_counter = 0;
        }

        self.time_point = now;
    }

    pub fn get_time(&self) -> f32 {
        self.time
    }

    pub fn get_delta_time(&self) -> f32 {
        self.delta_time
    }

    pub fn get_fps(&self) -> f32 {
        self.fps
    }
}

struct Input {
    keys: std::collections::HashMap<winit::keyboard::KeyCode, bool>,
}

impl Input {
    pub fn new() -> Self {
        Self {
            keys: std::collections::HashMap::new(),
        }
    }

    pub fn on_key_state_changed(&mut self, key: winit::keyboard::KeyCode, is_pressed: bool) {
        if let Some(value) = self.keys.get_mut(&key) {
            *value = is_pressed;
        } else {
            self.keys.insert(key, is_pressed);
        }
    }

    pub fn is_key_pressed(&self, key: winit::keyboard::KeyCode) -> bool {
        self.keys.get(&key).map_or(false, |v| *v)
    }
}

#[wasm_bindgen]
pub fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();

    let window = winit::window::WindowBuilder::new()
        .with_title("STARS-RS")
        .build(&event_loop).unwrap();
    let mut data = Vec::<u32>::with_capacity((window.inner_size().width * window.inner_size().height) as usize);

    let capacity = data.capacity();
    data.resize(capacity, 0);

    let mut stars = Stars::new();
    let mut timer = Timer::new();
    let mut input = Input::new();
    let mut frame = 0;

    stars.resize_stars(1024);
    stars.resize_surface(IVec2 { x: window.inner_size().width as i32, y: window.inner_size().height as i32 });

    let context = softbuffer::Context::new(&window).unwrap();
    let mut surface = softbuffer::Surface::new(&context, &window).unwrap();

    {
        let window_size = window.inner_size();
        let nonzero_size = std::num::NonZeroU32::new(window_size.width).zip(std::num::NonZeroU32::new(window_size.height)).unwrap();

        surface.resize(nonzero_size.0, nonzero_size.1).unwrap();
    }

    let mut resize_pressed_prev_frame = false;

    event_loop.run(|event, target| {
        match event {
            winit::event::Event::WindowEvent { window_id, event } => if window.id() == window_id {
                match event {
                    winit::event::WindowEvent::CloseRequested => {
                        target.exit()
                    }
                    winit::event::WindowEvent::KeyboardInput { event, .. } => if let winit::keyboard::PhysicalKey::Code(keycode) = event.physical_key {
                        input.on_key_state_changed(keycode, event.state == winit::event::ElementState::Pressed)
                    }
                    winit::event::WindowEvent::Resized(size) => {
                        let nzw = std::num::NonZeroU32::try_from(size.width).unwrap_or(std::num::NonZeroU32::try_from(1).unwrap());
                        let nzh = std::num::NonZeroU32::try_from(size.height).unwrap_or(std::num::NonZeroU32::try_from(1).unwrap());

                        _ = surface.resize(nzw, nzh);
                        stars.resize_surface(IVec2 { x: nzw.get() as i32, y: nzh.get() as i32 });
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        timer.response();

                        'fullscreen_resize: {
                            if input.is_key_pressed(winit::keyboard::KeyCode::F11) {
                                if resize_pressed_prev_frame {
                                    break 'fullscreen_resize;
                                }

                                if window.fullscreen().is_some() {
                                    window.set_fullscreen(None);
                                } else {
                                    // Find perfect suitable videomode
                                    if let Some(monitor) = window.current_monitor() {
                                        for mode in monitor.video_modes() {
                                            println!("{mode}");
                                        }
                                        let mut best_index: Option<usize> = None;
                                        let mut best_count: Option<u32> = None;
                                        for (index, count) in monitor.video_modes()
                                            .enumerate()
                                            .map(|(index, mode)|
                                                (index, (mode.bit_depth() == 32) as u32 + ((mode.refresh_rate_millihertz() == 48000) as u32 + (mode.size() == winit::dpi::PhysicalSize::new(640, 480)) as u32 * 2))
                                            ) {
                                            if Some(count) > best_count {
                                                best_count = Some(count);
                                                best_index = Some(index);
                                            }
                                        }

                                        if let Some(index) = best_index {
                                            window.set_fullscreen(Some(winit::window::Fullscreen::Exclusive(monitor.video_modes().nth(index).unwrap())));
                                        }
                                    }
                                }

                                resize_pressed_prev_frame = true;
                            } else {
                                resize_pressed_prev_frame = false;
                            }
                        }

                        {
                            type KeyCode = winit::keyboard::KeyCode;
                            let axis_y = (input.is_key_pressed(KeyCode::KeyS) as i32 - input.is_key_pressed(KeyCode::KeyW) as i32) as f32;
                            let axis_x = (input.is_key_pressed(KeyCode::KeyD) as i32 - input.is_key_pressed(KeyCode::KeyA) as i32) as f32;
                            let axis_s = (input.is_key_pressed(KeyCode::KeyR) as i32 - input.is_key_pressed(KeyCode::KeyF) as i32) as f32;
                            let axis_r = (input.is_key_pressed(KeyCode::KeyQ) as i32 - input.is_key_pressed(KeyCode::KeyE) as i32) as f32;

                            let axis_inv_len = 1.0 / (axis_x * axis_x + axis_y * axis_y).sqrt();

                            if !axis_inv_len.is_infinite() {
                                stars.translate(Vec3::new(
                                    axis_x * axis_inv_len * 5000.0 * timer.get_delta_time(),
                                    axis_y * axis_inv_len * 5000.0 * timer.get_delta_time(),
                                    0.0
                                ));
                            }

                            if axis_s.abs() > f32::EPSILON {
                                stars.speed = (stars.speed + axis_s * 5000.0 * timer.get_delta_time()).max(0.0);
                            }

                            if axis_r.abs() > f32::EPSILON {
                                stars.rotate_y(-axis_r * timer.get_delta_time());
                            }
                        }

                        if frame % 100 == 1 {
                            println!("{}", timer.get_fps());
                        }

                        stars.response(timer.get_delta_time());

                        let mut buffer = surface.buffer_mut().unwrap();
                        stars.render(unsafe {
                            std::slice::from_raw_parts_mut(buffer.as_mut_ptr(), buffer.len())
                        });
                        _ = buffer.present();

                        frame += 1;
                        window.request_redraw();
                    }
                    _ => {},
                }
            }
            _ => {},
        }
    }).unwrap();
}
