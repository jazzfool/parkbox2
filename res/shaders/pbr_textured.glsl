PBRMaterial material(Material params) {
	PBRMaterial pbr;

	uint albedo_id = params.textures[0];
	uint roughness_id = params.textures[1];
	uint metallic_id = params.textures[2];
	uint normal_id = params.textures[3];
	uint ao_id = params.textures[4];

	vec4 albedo = texture(textures[albedo_id], in_uv);
	vec4 roughness = texture(textures[roughness_id], in_uv);
	vec4 metallic = texture(textures[metallic_id], in_uv);
	vec3 normal = texture(textures[normal_id], in_uv).xyz;
	vec4 ao = texture(textures[ao_id], in_uv);

	albedo = pow(albedo, vec4(2.2));
	normal = get_normal_from_map(in_normal, normal);

	pbr.albedo = albedo.rgb;
	pbr.roughness = roughness.r;
	pbr.metallic = metallic.r;
	pbr.normal = normal;
	pbr.ao = ao.r;
	pbr.world_pos = in_position;
	pbr.reflectance = 0.04;

	return pbr;
}
